"""Agentic, tool-calling RAG loop with citation grounding.

The agent exposes four real tools backed by the Step-2 retriever:

* ``search_docs(query, k)``                — ranked, citable chunks
* ``lookup_asset(asset_id)``               — every document/revision for a machine
* ``get_spec_value(asset_id, spec_name)``  — parsed numeric value + unit + citation
* ``check_superseded(doc_id)``             — whether a newer revision exists

The loop is a ReAct-style controller (max ``max_tool_calls`` calls) that records
every step. A capable local model can drive the tool choice; when the model does
not emit a parseable action (typical for a 0.5B model) the controller falls back
to a deterministic plan so the loop always makes progress. The final answer is
synthesised from the accumulated context and must carry >= 1 citation or the
explicit sentinel ``INSUFFICIENT_EVIDENCE``.

A ``NaiveRAG`` baseline (single retrieval, no tools, no verifier) is provided for
the agentic-vs-naive comparison.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .backends import ExtractiveBackend, get_backend
from .chunking import Chunk
from .config import Settings, load_settings
from .corpus import ASSET_BY_ID
from .evaluate import _normalize
from .grounding import verify
from .retrieve import Hit, HybridRetriever, load_retriever

ASSET_RE = re.compile(r"\b([A-Z]{2,5}-\d{2})\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(N·m|N\.m|n·m|kN|PSI|psi|mm|µm|um|°C|C|RPM|rpm|%|h|hr|hours?|days?|CFH|Hz|m/s|µS/cm|L|kg|ton|A|V|W)"
)
REFUSAL = "INSUFFICIENT_EVIDENCE"

TOOL_DESCRIPTIONS = {
    "search_docs": "search_docs(query, k): ranked, citable document chunks for a query.",
    "lookup_asset": "lookup_asset(asset_id): every document and revision for a machine.",
    "get_spec_value": (
        "get_spec_value(asset_id, spec_name): the parsed numeric value, unit and citation "
        "for a named specification."
    ),
    "check_superseded": (
        "check_superseded(doc_id): whether a newer revision supersedes this document and "
        "what values changed."
    ),
}

_NUMERIC_WORDS = {
    "torque", "pressure", "temperature", "temp", "interval", "tolerance", "force", "rpm",
    "days", "day", "hours", "hour", "psi", "nm", "value", "how much", "how many", "setting",
    "concentration", "pre-charge", "precharge",
}
_CURRENT_WORDS = {
    "current", "today", "latest", "newest", "supersede", "supersedes", "superseded",
    "now", "updated", "replaced", "still", "effective",
}
# Words that are part of a question but carry no answerable content.
_QUESTION_STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "what", "which", "who", "whom", "whose",
    "how", "many", "much", "when", "where", "why", "does", "do", "did", "please", "tell",
    "show", "give", "list", "explain", "step", "steps", "taken", "take", "indicate",
    "mean", "means", "maximum", "minimum", "max", "min", "applied", "apply", "required",
    "require", "specified", "specify", "fires", "fire", "document", "documents",
    "difference", "differ", "current", "today", "latest", "newest", "now", "value",
    "values", "spec", "specs", "specification", "setting", "settings", "level", "above",
    "below", "under", "over", "between", "its", "this", "that", "these", "those",
    "their", "there", "northline", "precision", "machine", "machines", "manual",
    "manuals", "work", "instruction", "instructions", "revision", "rev", "set",
}
_ASSET_WORDS = {
    "cell", "cnc", "prs", "rob", "conv", "wash", "cmm", "laser", "weld", "grnd",
    "drll", "asm",
}
# Identity/administrative attributes that a maintenance corpus normally lacks.
_ATTRIBUTE_NOUNS = {
    "serial", "warranty", "firmware", "version", "manufacturer", "supplier", "price",
    "cost", "invoice", "purchase", "phone", "email", "contact", "installer", "number",
    "count", "quantity",
}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


# --------------------------------------------------------------------------- #
# Result objects                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    tool: str
    args: dict
    observation: str
    citations: list[str] = field(default_factory=list)
    n_chunks: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "args": self.args,
            "citations": self.citations,
            "n_chunks": self.n_chunks,
            "seconds": self.seconds,
            "observation": self.observation,
        }


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    backend: str = ""
    device: str = ""
    latency_s: float = 0.0
    gen_stats: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    n_context_chunks: int = 0
    mode: str = "agentic"

    @property
    def cited_doc_ids(self) -> list[str]:
        from .grounding import DOC_ID_RE

        ids: list[str] = []
        for cit in self.citations:
            m = DOC_ID_RE.search(cit)
            if m:
                ids.append(m.group(1))
        return ids

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "tool_calls": [c.as_dict() for c in self.tool_calls],
            "backend": self.backend,
            "device": self.device,
            "latency_s": self.latency_s,
            "gen_stats": self.gen_stats,
            "grounding": self.grounding,
            "n_context_chunks": self.n_context_chunks,
            "mode": self.mode,
        }


# --------------------------------------------------------------------------- #
# Tools                                                                       #
# --------------------------------------------------------------------------- #
class ShopFloorTools:
    """The four retrieval tools, backed by the hybrid retriever."""

    def __init__(self, retriever: HybridRetriever, settings: Settings | None = None) -> None:
        self.retriever = retriever
        self.settings = settings or load_settings()

    # -- 1. search ------------------------------------------------------ #
    def search_docs(
        self, query: str, k: int = 5, config: str = "hybrid_rrf_recency"
    ) -> tuple[str, list[Hit], list[str]]:
        hits = self.retriever.search(query, k=k, config=config)
        lines = [f"[{i + 1}] {h.citation} (score {h.score:.3f}){' [SUPERSEDED]' if h.superseded else ''}"
                 f"\n{h.chunk.text}" for i, h in enumerate(hits)]
        obs = "\n\n".join(lines) if lines else "(no results)"
        return obs, hits, [h.citation for h in hits]

    # -- 2. lookup_asset ------------------------------------------------ #
    def lookup_asset(self, asset_id: str) -> tuple[str, list[Hit], list[str]]:
        asset = asset_id.upper().strip()
        chunks = [c for c in self.retriever.chunks if c.asset == asset]
        if not chunks:
            return f"(no documents found for asset {asset})", [], []
        inventory: dict[str, Chunk] = {}
        for c in chunks:
            inventory.setdefault(c.doc_id, c)
        rows = [
            f"{c.doc_id} (rev {c.revision}, {c.effective_date}, {c.doc_type})"
            for c in sorted(inventory.values(), key=lambda c: c.doc_id)
        ]
        obs = f"Documents for {asset}:\n" + "\n".join(rows)
        hits = [Hit(chunk=c, score=0.0) for c in chunks]
        citations = [c.citation() for c in inventory.values()]
        return obs, hits, citations

    # -- 3. get_spec_value ---------------------------------------------- #
    def get_spec_value(self, asset_id: str, spec_name: str) -> tuple[str, list[Hit], list[str]]:
        asset = asset_id.upper().strip()
        want = {t for t in _TOKEN_RE.findall(spec_name.lower())}
        best: tuple[float, str, Chunk] | None = None
        for c in self.retriever.chunks:
            if c.asset != asset:
                continue
            for line in c.text.splitlines():
                m = _UNIT_RE.search(line)
                if not m:
                    continue
                line_tokens = set(_TOKEN_RE.findall(line.lower()))
                overlap = len(want & line_tokens)
                if overlap == 0:
                    continue
                score = overlap + (0.5 if c.doc_type in ("manual", "spec") else 0.0)
                if best is None or score > best[0]:
                    best = (score, line.strip(), c)
        if best is None:
            return (
                f"(no numeric spec named {spec_name!r} found for {asset})",
                [],
                [],
            )
        _, line, chunk = best
        m = _UNIT_RE.search(line)
        value, unit = (m.group(1), m.group(2)) if m else ("", "")
        obs = f"{asset} {line}\nvalue={value} unit={unit}\ncitation={chunk.citation()}"
        return obs, [Hit(chunk=chunk, score=0.0)], [chunk.citation()]

    # -- 4. check_superseded -------------------------------------------- #
    def check_superseded(self, doc_id: str) -> tuple[str, list[Hit], list[str]]:
        target = doc_id.strip()
        superseders = [c for c in self.retriever.chunks if str(c.supersedes or "") == target]
        if not superseders:
            return f"{target}: not superseded (no newer revision on record).", [], []
        by: dict[str, Chunk] = {}
        for c in superseders:
            by.setdefault(c.doc_id, c)
        obs = f"{target} IS superseded by: " + ", ".join(
            f"{c.doc_id} (rev {c.revision}, {c.effective_date})" for c in by.values()
        )
        hits = [Hit(chunk=c, score=0.0) for c in superseders]
        citations = [c.citation() for c in by.values()]
        return obs, hits, citations

    # -- langchain_core wrappers ---------------------------------------- #
    def langchain_tools(self) -> list:
        """Expose the tools to LangChain (best-effort; empty if unavailable)."""
        try:
            from langchain_core.tools import Tool
        except Exception:  # pragma: no cover - langchain optional in edge envs
            return []

        def _search(payload: str) -> str:
            args = _json_args(payload)
            return self.search_docs(args.get("query", payload), int(args.get("k", 5)))[0]

        def _lookup(payload: str) -> str:
            return self.lookup_asset(str(_json_args(payload).get("asset_id", payload)))[0]

        def _spec(payload: str) -> str:
            a = _json_args(payload)
            return self.get_spec_value(str(a.get("asset_id", "")), str(a.get("spec_name", "")))[0]

        def _sup(payload: str) -> str:
            return self.check_superseded(str(_json_args(payload).get("doc_id", payload)))[0]

        return [
            Tool(name="search_docs", func=_search, description=TOOL_DESCRIPTIONS["search_docs"]),
            Tool(name="lookup_asset", func=_lookup, description=TOOL_DESCRIPTIONS["lookup_asset"]),
            Tool(name="get_spec_value", func=_spec, description=TOOL_DESCRIPTIONS["get_spec_value"]),
            Tool(name="check_superseded", func=_sup, description=TOOL_DESCRIPTIONS["check_superseded"]),
        ]


def _json_args(payload: str) -> dict:
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Prompt helpers                                                              #
# --------------------------------------------------------------------------- #
def build_context_block(chunks: list[Chunk]) -> str:
    lines = ["### CONTEXT"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c.citation()}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)


def synth_prompt(question: str, chunks: list[Chunk]) -> str:
    return (
        "You are the Northline Precision Cell 4 shop-floor assistant.\n"
        "Answer the technician's question using ONLY the context below.\n"
        "Cite the document ids you used, e.g. (MANUAL-CNC-07 §4).\n"
        "If the answer is not present in the context, reply exactly INSUFFICIENT_EVIDENCE.\n"
        "Prefer the newest applicable revision when documents conflict.\n\n"
        f"{build_context_block(chunks)}\n"
        f"### QUESTION\n{question}\n\n"
        "### INSTRUCTIONS\nAnswer in 1-3 sentences and end with the citation(s).\n"
    )


def react_prompt(question: str, scratchpad: str) -> str:
    tools = "\n".join(f"- {v}" for v in TOOL_DESCRIPTIONS.values())
    return (
        "You are an agent with tools. Choose the next tool or finish.\n"
        "Respond in EXACTLY one of these two forms:\n"
        'Action: <tool_name>\nAction Input: <json>\n'
        "or\n"
        "Final Answer: <answer with citations>\n\n"
        f"Tools:\n{tools}\n\n"
        f"### QUESTION\n{question}\n\n"
        f"### SCRATCHPAD\n{scratchpad or '(empty)'}\n"
    )


def _parse_react(text: str) -> tuple[str, object, dict] | None:
    if not text:
        return None
    if "Final Answer:" in text:
        return ("final", text.split("Final Answer:", 1)[1].strip(), {})
    m = re.search(r"Action:\s*([a-z_]+)\s*\n?\s*Action Input:\s*(\{.*?\})", text, re.S | re.I)
    if m:
        tool = m.group(1).strip()
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        if tool in TOOL_DESCRIPTIONS:
            return ("action", tool, args)
    return None


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class Agent:
    """ReAct-style agentic RAG controller."""

    def __init__(
        self,
        retriever: HybridRetriever,
        backend: object | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.retriever = retriever
        self.backend = backend or get_backend(self.settings.backend, self.settings)
        self.tools = ShopFloorTools(retriever, self.settings)
        self.use_llm = not isinstance(self.backend, ExtractiveBackend)
        self._corpus_vocab = self._build_corpus_vocab()

    def _build_corpus_vocab(self) -> set[str]:
        vocab: set[str] = set()
        for c in self.retriever.chunks:
            for t in re.findall(r"[a-z]+", c.text.lower()):
                if len(t) >= 3:
                    vocab.add(_stem(t))
        return vocab

    def answerability(self, question: str, chunks: list[Chunk]) -> tuple[bool, str]:
        """Heuristic answerability check: does the corpus/context even cover the ask?"""
        tokens = re.findall(r"[a-z]+", question.lower())
        content = {
            _stem(t)
            for t in tokens
            if len(t) >= 3 and t not in _QUESTION_STOPWORDS and t not in _ASSET_WORDS
        }
        absent = sorted(t for t in content if t not in self._corpus_vocab)
        if absent:
            return False, f"question terms absent from the corpus: {absent}"
        ctx_vocab: set[str] = set()
        for c in chunks:
            for t in re.findall(r"[a-z]+", c.text.lower()):
                ctx_vocab.add(_stem(t))
        asked = {t for t in tokens if t in _ATTRIBUTE_NOUNS}
        if asked:
            entities = {
                _stem(t)
                for t in tokens
                if len(t) >= 3
                and t not in _QUESTION_STOPWORDS
                and t not in _ASSET_WORDS
                and t not in _ATTRIBUTE_NOUNS
            }
            for attr in sorted(asked):
                attr_stem = _stem(attr)
                found = False
                for c in chunks:
                    for line in c.text.splitlines():
                        line_tokens = {_stem(x) for x in re.findall(r"[a-z]+", line.lower())}
                        if attr_stem in line_tokens and (entities & line_tokens):
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return (
                        False,
                        f"attribute '{attr}' is not co-located with the requested entity "
                        "anywhere in the retrieved context",
                    )
        return True, ""

    # ------------------------------------------------------------------ #
    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    @property
    def device(self) -> str:
        return getattr(self.backend, "device", "")

    # ------------------------------------------------------------------ #
    @staticmethod
    def detect_asset(question: str) -> str | None:
        for m in ASSET_RE.findall(question.upper()):
            if m in ASSET_BY_ID:
                return m
        return None

    @staticmethod
    def wants_current(question: str) -> bool:
        q = question.lower()
        return any(w in q for w in _CURRENT_WORDS)

    @staticmethod
    def is_numeric(question: str) -> bool:
        q = question.lower()
        if re.search(r"\d", q):
            return True
        return any(w in q for w in _NUMERIC_WORDS)

    def plan(self, question: str) -> list[tuple[str, dict]]:
        """Deterministic tool plan used as the controller's fallback."""
        steps: list[tuple[str, dict]] = [("search_docs", {"query": question, "k": self.settings.top_k})]
        # A fault code is a precise key: refine the search with the bare code.
        fault = re.search(r"\bE-\d{3}\b", question.upper())
        if fault:
            # An exact fault code is a keyword lookup: use BM25, which is precise
            # for rare codes and not distracted by semantic similarity.
            steps.append(
                ("search_docs", {"query": fault.group(0), "k": self.settings.top_k, "config": "bm25_only"})
            )
        asset = self.detect_asset(question)
        if asset:
            steps.append(("lookup_asset", {"asset_id": asset}))
            if self.is_numeric(question):
                steps.append(("get_spec_value", {"asset_id": asset, "spec_name": question}))
            if self.wants_current(question) or self.is_numeric(question):
                steps.append(("check_superseded", {"doc_id": f"MANUAL-{asset}"}))
        return steps[: self.settings.max_tool_calls]

    # ------------------------------------------------------------------ #
    def _execute(self, tool: str, args: dict) -> tuple[ToolCall, list[Hit], list[str]]:
        t0 = time.perf_counter()
        if tool == "search_docs":
            obs, hits, cites = self.tools.search_docs(
                args.get("query", ""),
                int(args.get("k", self.settings.top_k)),
                str(args.get("config", "hybrid_rrf_recency")),
            )
        elif tool == "lookup_asset":
            obs, hits, cites = self.tools.lookup_asset(str(args.get("asset_id", "")))
        elif tool == "get_spec_value":
            obs, hits, cites = self.tools.get_spec_value(
                str(args.get("asset_id", "")), str(args.get("spec_name", ""))
            )
        elif tool == "check_superseded":
            obs, hits, cites = self.tools.check_superseded(str(args.get("doc_id", "")))
        else:
            obs, hits, cites = f"(unknown tool {tool})", [], []
        sec = time.perf_counter() - t0
        return ToolCall(tool=tool, args=args, observation=obs, citations=cites,
                        n_chunks=len(hits), seconds=round(sec, 4)), hits, cites

    # ------------------------------------------------------------------ #
    def run(self, question: str, show_trace: bool = False) -> AgentResult:
        t0 = time.perf_counter()
        ctx: list[Hit] = []
        tool_calls: list[ToolCall] = []
        transcript: list[dict] = []
        gen_stats: list[dict] = []
        plan = self.plan(question)
        plan_idx = 0
        scratchpad = ""
        final_from_llm: str | None = None
        llm_active = self.use_llm

        for step in range(self.settings.max_tool_calls):
            choice: tuple[str, object, dict] | None = None
            if llm_active:
                try:
                    gen = self.backend.generate(  # type: ignore[attr-defined]
                        react_prompt(question, scratchpad), max_tokens=96, stream=False
                    )
                    gen_stats.append(gen.stats.as_dict())
                    choice = _parse_react(gen.text)
                    transcript.append({"step": step, "llm": gen.text})
                except Exception as exc:  # pragma: no cover - backend dependent
                    transcript.append({"step": step, "llm_error": str(exc)})
                    choice = None
                if choice and choice[0] == "final":
                    final_from_llm = str(choice[1])
                    break
                if choice is None:
                    # Small local models often cannot emit a valid action; stop
                    # asking and let the deterministic controller finish.
                    llm_active = False
                    transcript.append({"step": step, "llm_tool_selection": "fallback_to_plan"})

            if choice and choice[0] == "action":
                tool, args = str(choice[1]), dict(choice[2])  # type: ignore[arg-type]
            else:
                if plan_idx >= len(plan):
                    break
                tool, args = plan[plan_idx]
                plan_idx += 1

            call, hits, _ = self._execute(tool, args)
            tool_calls.append(call)
            ctx.extend(hits)
            scratchpad += (
                f"Action: {tool}\nAction Input: {json.dumps(args)}\n"
                f"Observation: {call.observation}\n"
            )
            transcript.append({"step": step, "tool": tool, "args": args,
                               "n_chunks": call.n_chunks, "seconds": call.seconds})
            if not llm_active and plan_idx >= len(plan):
                break

        answer, citations, refused = self._synthesize(question, ctx, final_from_llm, gen_stats)
        ok, reason = self.answerability(question, _dedupe(ctx))
        if not ok:
            answer, citations, refused = REFUSAL, [], True
        grounding = verify(answer, _dedupe(ctx), cited_doc_ids=_doc_ids(citations))
        latency = time.perf_counter() - t0

        return AgentResult(
            question=question,
            answer=answer,
            citations=citations,
            refused=refused,
            refusal_reason=reason if not ok else "",
            tool_calls=tool_calls,
            transcript=transcript,
            backend=self.backend_name,
            device=self.device,
            latency_s=round(latency, 4),
            gen_stats=gen_stats,
            grounding=grounding.as_dict(),
            n_context_chunks=len(_dedupe(ctx)),
            mode="agentic",
        )

    # ------------------------------------------------------------------ #
    def _synthesize(
        self,
        question: str,
        ctx: list[Hit],
        final_from_llm: str | None,
        gen_stats: list[dict],
    ) -> tuple[str, list[str], bool]:
        chunks = _ordered(_dedupe(ctx))
        if final_from_llm is not None and final_from_llm.strip():
            answer = final_from_llm.strip()
            citations = _derive_citations(answer, chunks)
            refused = REFUSAL in answer.upper() or (not citations and not chunks)
            if refused:
                return REFUSAL, [], True
            return answer, citations, False

        prompt = synth_prompt(question, chunks)
        try:
            gen = self.backend.generate(prompt, max_tokens=self.settings.max_new_tokens, stream=True)  # type: ignore[attr-defined]
            gen_stats.append(gen.stats.as_dict())
            answer = gen.text.strip()
        except Exception as exc:  # pragma: no cover
            return f"backend error: {exc}", [], True

        if REFUSAL in answer.upper() or not answer:
            return REFUSAL, [], True
        citations = _derive_citations(answer, chunks)
        if not citations and chunks and _normalize(answer)[:40] not in " ".join(
            _normalize(c.text) for c in chunks
        ):
            # Nothing in the answer is traceable to context -> treat as refusal.
            return REFUSAL, [], True
        return answer, citations, False


# --------------------------------------------------------------------------- #
# Naive baseline                                                              #
# --------------------------------------------------------------------------- #
class NaiveRAG:
    """Single retrieval, no tools, no verifier — the baseline to beat."""

    def __init__(self, retriever: HybridRetriever, backend: object | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.retriever = retriever
        self.backend = backend or get_backend(self.settings.backend, self.settings)
        self.tools = ShopFloorTools(retriever, self.settings)

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    @property
    def device(self) -> str:
        return getattr(self.backend, "device", "")

    def run(self, question: str, show_trace: bool = False) -> AgentResult:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=self.settings.top_k, config="hybrid_rrf_recency")
        chunks = _ordered([h.chunk for h in hits])
        gen_stats: list[dict] = []
        prompt = synth_prompt(question, chunks)
        try:
            gen = self.backend.generate(prompt, max_tokens=self.settings.max_new_tokens, stream=True)  # type: ignore[attr-defined]
            gen_stats.append(gen.stats.as_dict())
            answer = gen.text.strip()
        except Exception as exc:  # pragma: no cover
            answer = f"backend error: {exc}"
        if REFUSAL in answer.upper() or not answer:
            answer, citations, refused = REFUSAL, [], True
        else:
            citations = _derive_citations(answer, chunks)
            refused = False
        latency = time.perf_counter() - t0
        return AgentResult(
            question=question,
            answer=answer,
            citations=citations,
            refused=refused,
            tool_calls=[],
            transcript=[],
            backend=self.backend_name,
            device=self.device,
            latency_s=round(latency, 4),
            gen_stats=gen_stats,
            grounding={},
            n_context_chunks=len(chunks),
            mode="naive",
        )


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _dedupe(hits: list[Hit]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for h in hits:
        if h.chunk.chunk_id in seen:
            continue
        seen.add(h.chunk.chunk_id)
        out.append(h.chunk)
    return out


def _ordered(chunks: list[Chunk]) -> list[Chunk]:
    """Newest effective_date first, so synthesis prefers the current revision."""
    return sorted(chunks, key=lambda c: c.effective_date or "", reverse=True)


def _doc_ids(citations: list[str]) -> list[str]:
    from .grounding import DOC_ID_RE

    ids: list[str] = []
    for c in citations:
        m = DOC_ID_RE.search(c)
        if m:
            ids.append(m.group(1))
    return ids


def _derive_citations(answer: str, chunks: list[Chunk]) -> list[str]:
    """Map an answer back to the context chunks it came from."""
    if not chunks:
        return []
    norm_answer = _normalize(answer)
    answer_lower = answer.lower()
    ordered: list[str] = []
    seen: set[str] = set()

    # 1. documents explicitly named in the answer (case-insensitive)
    for c in chunks:
        if c.doc_id.lower() in answer_lower and c.doc_id not in seen:
            seen.add(c.doc_id)
            ordered.append(c.citation())

    # 2. text containment (extractive answers copy spans verbatim)
    head = norm_answer[:48]
    for c in chunks:
        body = _normalize(c.text)
        if head and head in body and c.doc_id not in seen:
            seen.add(c.doc_id)
            ordered.append(c.citation())

    # 3. fallback: the chunk with the greatest word overlap with the answer
    if not ordered:
        ans_tokens = set(_TOKEN_RE.findall(answer.lower()))
        best = max(
            chunks,
            key=lambda c: len(ans_tokens & set(_TOKEN_RE.findall(c.text.lower()))),
        )
        ordered.append(best.citation())
    return ordered


# --------------------------------------------------------------------------- #
# Convenience                                                                 #
# --------------------------------------------------------------------------- #
def load_agent(
    index_dir: str = "artifacts/index",
    backend_name: str | None = None,
    settings: Settings | None = None,
    retriever: HybridRetriever | None = None,
):
    settings = settings or load_settings()
    retriever = retriever or load_retriever(index_dir, settings)
    backend = get_backend(backend_name or settings.backend, settings)
    return Agent(retriever, backend, settings)


if __name__ == "__main__":  # pragma: no cover
    agent = load_agent(backend_name="extractive")
    res = agent.run("What is the current torque spec for the CNC-07 spindle clamp bolts?")
    print(res.answer)
    print(res.citations)
