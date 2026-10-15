"""Post-hoc grounding verifier.

Given an answer and the chunks that were actually retrieved, check that:

* every cited document id was really retrieved (no fabricated citations), and
* every numeric value in the answer appears in a retrieved chunk
  (no fabricated numbers).

The verifier never edits the answer; it only reports what is ungrounded, so the
caller can decide how strict to be.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .chunking import Chunk
from .evaluate import _normalize

DOC_ID_RE = re.compile(r"\b((?:MANUAL|SPEC|QC-WI|INC)-[A-Z0-9-]+)\b")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class GroundingReport:
    grounded: bool
    cited_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)
    checked_numbers: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _reference_text(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for c in chunks:
        parts.append(c.text)
        parts.append(c.citation())
        if c.title:
            parts.append(c.title)
    return _normalize(" \n ".join(parts))


def verify(
    answer: str,
    retrieved_chunks: list[Chunk],
    cited_doc_ids: list[str] | None = None,
) -> GroundingReport:
    """Check citations and numbers in ``answer`` against ``retrieved_chunks``."""
    retrieved_doc_ids = sorted({c.doc_id for c in retrieved_chunks})
    referenced = sorted(set(DOC_ID_RE.findall(answer or "")))
    cited = sorted(set(cited_doc_ids or []) | set(referenced))

    ungrounded_citations = [d for d in cited if d not in set(retrieved_doc_ids)]

    reference = _reference_text(retrieved_chunks)
    checked: list[str] = []
    ungrounded_numbers: list[str] = []
    for num in NUMBER_RE.findall(answer or ""):
        # Skip numbers that only appear inside a citation (e.g. docs "2026-001").
        stripped = num.strip(".").rstrip(".")
        if not stripped:
            continue
        checked.append(stripped)
        if _normalize(stripped) not in reference:
            ungrounded_numbers.append(stripped)

    return GroundingReport(
        grounded=not ungrounded_citations and not ungrounded_numbers,
        cited_doc_ids=cited,
        retrieved_doc_ids=retrieved_doc_ids,
        ungrounded_citations=ungrounded_citations,
        checked_numbers=checked,
        ungrounded_numbers=ungrounded_numbers,
    )
