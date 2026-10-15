"""Pluggable local LLM backends.

Three interchangeable backends implement the same tiny interface
(:class:`LLMBackend`) so the agent does not care where the tokens come from:

* :class:`TransformersBackend` — Hugging Face ``transformers`` (CPU or CUDA),
  fp16 on CUDA / fp32 on CPU, with a real ``TextIteratorStreamer`` TTFT.
* :class:`OpenAICompatBackend` — any OpenAI-compatible ``/v1/chat/completions``
  endpoint, so a vLLM server or an NVIDIA NIM container drops straight in.
* :class:`ExtractiveBackend` — no weights; composes an answer from retrieved
  spans. Deterministic and offline, so CI and unit tests need no download.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from .config import Settings, load_settings


# --------------------------------------------------------------------------- #
# Value objects                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class GenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float | None = None
    total_s: float = 0.0
    tokens_per_s: float | None = None
    device: str = ""
    model: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Generation:
    text: str
    stats: GenStats = field(default_factory=GenStats)


# --------------------------------------------------------------------------- #
# Extractive backend (offline, deterministic)                                 #
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "is", "are", "what", "which",
    "who", "how", "does", "do", "and", "or", "at", "by", "with", "that", "this",
    "cell", "northline", "please", "spec", "specification", "value", "required",
    "must", "should", "be", "it", "as", "if", "i", "we", "you", "mean", "means",
}
_HISTORICAL = {
    "was", "were", "previous", "previously", "insufficient", "old", "superseded",
    "originally", "no", "longer", "changed", "from", "before", "used",
}
_LIST_SEP_RE = re.compile(r"^\|?\s*[-:| ]+\s*$")
_UNIT_RE = re.compile(
    r"\d[\d.,]*\s*(?:\|\s*)?"
    r"(?:N·m|N\.m|kN|N\b|PSI|psi|mm|µm|°C|RPM|rpm|%|CFH|Hz|m/s|ms|µS/cm|kg|ton|days?|h\b|L\b|V\b)"
)
_EVERY_RE = re.compile(r"\b(?:every|per)\s+\d")
_ASSET_TOKEN_RE = re.compile(r"\b([a-z]{2,5}-\d{2})\b")
_NUMERIC_HINT = {
    "torque", "pressure", "temperature", "interval", "tolerance", "force", "rpm",
    "days", "hours", "psi", "nm", "concentration", "charge", "speed", "number",
    "serial", "firmware", "version", "warranty",
}


def _content_tokens(text: str, strip_assets: bool = False) -> list[str]:
    """Keep alphabetic tokens >= 3 chars and numeric tokens."""
    if strip_assets:
        text = _ASSET_TOKEN_RE.sub(" ", text.lower())
    out: list[str] = []
    for tok in re.findall(r"[a-z]+|\d+", text.lower()):
        if tok.isdigit() or len(tok) >= 3:
            out.append(tok)
    return out


class ExtractiveBackend:
    """Compose an answer by selecting the most relevant retrieved sentences."""

    name = "extractive"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.model_id = "extractive-rule-based"
        self.device = "cpu"
        self.available = True
        self.load_error: str | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_context(prompt: str) -> list[tuple[str, str]]:
        """Extract ``(citation, text)`` pairs from the agent's prompt context."""
        if "### CONTEXT" not in prompt:
            # Fall back: treat the whole prompt body after QUESTION as context.
            return []
        block = prompt.split("### CONTEXT", 1)[1]
        for marker in ("### INSTRUCTIONS", "### QUESTION", "### FINAL"):
            if marker in block:
                block = block.split(marker, 1)[0]
        entries: list[tuple[str, str]] = []
        # Entries look like:  [1] <citation>\n<text>
        for match in re.finditer(r"\[(\d+)\]\s*(.+?)\n(.*?)(?=\n\[\d+\]|\Z)", block, re.S):
            citation = match.group(2).strip()
            text = match.group(3).strip()
            entries.append((citation, text))
        return entries

    def _question(self, prompt: str) -> str:
        if "### QUESTION" in prompt:
            q = prompt.split("### QUESTION", 1)[1]
            for marker in ("### CONTEXT", "### INSTRUCTIONS", "### FINAL"):
                if marker in q:
                    q = q.split(marker, 1)[0]
            return q.strip()
        return prompt.strip()

    def generate(self, prompt: str, max_tokens: int = 200, stream: bool = False) -> Generation:
        t0 = time.perf_counter()
        question = self._question(prompt)
        entries = self._parse_context(prompt)
        q_tokens = {t for t in _content_tokens(question, strip_assets=True) if t not in _STOPWORDS}
        question_wo_asset = _ASSET_TOKEN_RE.sub(" ", question.lower())
        wants_number = bool(q_tokens & _NUMERIC_HINT) or bool(re.search(r"\d", question_wo_asset))
        asset_match = _ASSET_TOKEN_RE.search(question.lower())
        asset_tok = asset_match.group(1) if asset_match else None

        scored: list[tuple[float, str, str]] = []
        for citation, text in entries:
            cit_norm = citation.lower()
            for sent in re.split(r"(?<=[.;])\s+|\n+", text):
                sent = sent.strip()
                if not sent or _LIST_SEP_RE.match(sent):
                    continue
                if sent.startswith("#"):
                    continue  # document heading
                s_tokens = set(_content_tokens(sent, strip_assets=True))
                overlap = q_tokens & s_tokens
                if not any(t.isdigit() or len(t) >= 4 for t in overlap):
                    continue
                has_value = bool(_UNIT_RE.search(sent) or _EVERY_RE.search(sent))
                if wants_number and not has_value:
                    continue
                units = len(_UNIT_RE.findall(sent))
                score = float(len(overlap))
                # An exact numeric code shared with the question (fault code,
                # section number) is a strong signal.
                score += 2.0 * sum(1 for t in overlap if t.isdigit())
                if sent.startswith("|") and wants_number:
                    score += 2.2  # a spec/change table row carries the value
                if units >= 2:
                    score += 1.8  # an explicit old->new change row is authoritative
                if asset_tok and (asset_tok in cit_norm or asset_tok in sent.lower()):
                    score += 0.8
                if sent.startswith("**"):
                    score -= 0.8  # document metadata line
                if {w for w in re.findall(r"[a-z]+", sent.lower())} & _HISTORICAL:
                    score -= 1.2
                score -= 0.01 * len(sent)
                if score > 0:
                    scored.append((score, citation, sent))

        if not scored:
            answer = "INSUFFICIENT_EVIDENCE"
        else:
            scored.sort(key=lambda s: -s[0])
            best = scored[:4]
            chosen = [best[0]]
            seen_text = {best[0][2]}
            # Add a second sentence only from the same document, to keep the
            # citation list precise.
            for cand in best[1:]:
                if cand[1] == chosen[0][1] and cand[2] not in seen_text:
                    chosen.append(cand)
                    seen_text.add(cand[2])
                    break
            answer = " ".join(c[2] for c in chosen)

        elapsed = time.perf_counter() - t0
        n_completion = len(re.findall(r"\S+", answer))
        stats = GenStats(
            prompt_tokens=len(re.findall(r"\S+", prompt)),
            completion_tokens=n_completion,
            ttft_s=round(elapsed * 0.5, 6),
            total_s=round(elapsed, 6),
            tokens_per_s=round(n_completion / elapsed, 3) if elapsed > 0 else None,
            device="cpu",
            model=self.model_id,
        )
        return Generation(text=answer, stats=stats)


# --------------------------------------------------------------------------- #
# Hugging Face Transformers backend                                           #
# --------------------------------------------------------------------------- #
class TransformersBackend:
    """Local HF transformers generation with a real streaming TTFT."""

    name = "transformers"

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        device_map: str | None = None,
        max_new_tokens: int = 320,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.model_id = model_id or self.settings.llm_model
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.device = "cpu"
        self.available = False
        self.load_error: str | None = None
        self._model = None
        self._tokenizer = None
        self._init_model(device)

    def _init_model(self, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            kwargs: dict = {}
            if self.device_map:
                kwargs["device_map"] = self.device_map
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, dtype=dtype, **kwargs
                )
            except TypeError:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype=dtype, **kwargs
                )
            if not self.device_map:
                self._model.to(device)
            self._model.eval()
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self.available = True
        except Exception as exc:  # pragma: no cover - network/host dependent
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.available = False

    # ------------------------------------------------------------------ #
    def _chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    def generate(self, prompt: str, max_tokens: int | None = None, stream: bool = True) -> Generation:
        import torch

        if not self.available:
            raise RuntimeError(f"transformers backend unavailable: {self.load_error}")
        max_new = max_tokens or self.max_new_tokens
        text = self._chat(prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new,
            do_sample=self.settings.temperature > 0,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        if self.settings.temperature > 0:
            gen_kwargs["temperature"] = max(0.01, self.settings.temperature)

        t0 = time.perf_counter()
        ttft: float | None = None
        if stream:
            from threading import Thread

            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            thread = Thread(target=self._model.generate, kwargs={**gen_kwargs, "streamer": streamer})
            thread.start()
            pieces: list[str] = []
            for token in streamer:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                pieces.append(token)
            thread.join()
            output_text = "".join(pieces)
        else:
            with torch.no_grad():
                out = self._model.generate(**gen_kwargs)
            output_text = self._tokenizer.decode(
                out[0][prompt_tokens:], skip_special_tokens=True
            )
        total = time.perf_counter() - t0
        completion_tokens = len(self._tokenizer(output_text)["input_ids"])
        gen_time = (total - ttft) if ttft is not None else total
        stats = GenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=round(ttft, 6) if ttft is not None else None,
            total_s=round(total, 6),
            tokens_per_s=round(completion_tokens / gen_time, 3) if gen_time > 0 else None,
            device=self.device,
            model=self.model_id,
        )
        return Generation(text=output_text.strip(), stats=stats)


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoint backend (vLLM / NVIDIA NIM)                      #
# --------------------------------------------------------------------------- #
class OpenAICompatBackend:
    """Talk to any OpenAI-compatible ``/v1/chat/completions`` server."""

    name = "openai_compat"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.endpoint = (endpoint if endpoint is not None else self.settings.endpoint).rstrip("/")
        self.api_key = api_key if api_key is not None else self.settings.api_key
        self.model_id = model_id or self.settings.llm_model
        self.device = "remote"
        self.load_error: str | None = None
        self.available = bool(self.endpoint)

    def _url(self) -> str:
        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint}/chat/completions"
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        return f"{self.endpoint}/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 320, stream: bool = True) -> Generation:
        import requests

        if not self.available:
            raise RuntimeError("no endpoint configured (set LOCALRAG_ENDPOINT)")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.settings.temperature,
            "stream": bool(stream),
        }
        t0 = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        usage: dict = {}
        if stream:
            with requests.post(self._url(), json=payload, headers=headers, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        text_parts.append(delta)
            text = "".join(text_parts)
        else:
            resp = requests.post(self._url(), json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            obj = resp.json()
            text = obj["choices"][0]["message"]["content"]
            usage = obj.get("usage", {})
        total = time.perf_counter() - t0
        completion_tokens = usage.get("completion_tokens") or len(text.split())
        gen_time = (total - ttft) if ttft is not None else total
        stats = GenStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            ttft_s=round(ttft, 6) if ttft is not None else None,
            total_s=round(total, 6),
            tokens_per_s=round(completion_tokens / gen_time, 3) if gen_time > 0 else None,
            device="remote",
            model=self.model_id,
        )
        return Generation(text=text.strip(), stats=stats)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
BACKEND_NAMES = ("auto", "transformers", "openai_compat", "extractive")


def get_backend(name: str = "auto", settings: Settings | None = None, **kwargs) -> object:
    """Return a backend instance.

    ``auto`` prefers an OpenAI-compatible endpoint when one is configured, then
    the local transformers model when it loads, and finally the extractive
    backend so the demo never hard-fails.
    """
    settings = settings or load_settings()
    name = (name or "auto").lower()
    if name not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")

    if name == "extractive":
        return ExtractiveBackend(settings)
    if name == "openai_compat":
        return OpenAICompatBackend(settings=settings, **kwargs)
    if name == "transformers":
        return TransformersBackend(settings=settings, **kwargs)

    # auto
    if settings.endpoint:
        return OpenAICompatBackend(settings=settings)
    backend = TransformersBackend(settings=settings, **kwargs)
    if getattr(backend, "available", False):
        return backend
    return ExtractiveBackend(settings)


def backend_status(settings: Settings | None = None) -> list[dict]:
    """Report which backends are usable on this host (without heavy loading)."""
    settings = settings or load_settings()
    status: list[dict] = []

    import_ok, import_err = True, None
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        import_ok, import_err = False, f"{type(exc).__name__}: {exc}"
    status.append(
        {
            "backend": "transformers",
            "available": import_ok,
            "detail": (
                f"model={settings.llm_model} device={'cuda' if _cuda() else 'cpu'} "
                "(weights download on first use)"
                if import_ok
                else import_err
            ),
        }
    )
    status.append(
        {
            "backend": "openai_compat",
            "available": bool(settings.endpoint),
            "detail": settings.endpoint or "set LOCALRAG_ENDPOINT to a vLLM/NIM /v1 endpoint",
        }
    )
    status.append({"backend": "extractive", "available": True, "detail": "offline, no weights"})
    return status


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover
    for row in backend_status():
        print(row)
