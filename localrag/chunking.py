"""Section-aware Markdown chunker.

Splits each document on Markdown ``##`` headings so that a chunk maps cleanly to
a citable section (``MANUAL-CNC-07 §4``). Tables are never split across chunks:
over-long sections are further split only on blank-line paragraph boundaries,
and a block that starts with a table header row is kept whole.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

_HEADING_RE = re.compile(r"^(#{2})\s+(.*)$")
_SECTION_CODE_RE = re.compile(r"^§\s*([0-9A-Za-z.]+)\s*(.*)$")


@dataclass
class Chunk:
    """A single retrievable, citable unit of a document."""

    chunk_id: str
    doc_id: str
    section: str  # short code, e.g. "§4"
    section_title: str  # e.g. "Torque Specs"
    text: str
    revision: str
    effective_date: str
    char_span: list[int] = field(default_factory=lambda: [0, 0])
    doc_type: str = ""
    asset: str = ""
    title: str = ""
    supersedes: str | None = None
    supersedes_section: str | None = None

    # ------------------------------------------------------------------ #
    def citation(self) -> str:
        sec = f" {self.section}" if self.section else ""
        return f"{self.doc_id}{sec} (rev {self.revision}, {self.effective_date})"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


# --------------------------------------------------------------------------- #
def parse_document(path: str | Path) -> tuple[dict, str]:
    """Return ``(front_matter, body)`` for a Markdown document with YAML front matter."""
    text = Path(path).read_text(encoding="utf-8")
    meta: dict = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip("\n")
    return meta, body


def _split_long_section(text: str, size: int, overlap: int) -> list[str]:
    """Split an over-long section on blank lines, keeping table blocks intact."""
    if len(text) <= size:
        return [text]

    # Group lines into blocks separated by blank lines.
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for block in blocks:
        block_text = "\n".join(block)
        block_len = len(block_text) + 1
        if buf and buf_len + block_len > size and not _is_table(block):
            chunks.append("\n\n".join(buf).strip())
            # carry a small overlap from the tail of the previous chunk
            tail = "\n\n".join(buf)[-overlap:] if overlap else ""
            buf = [tail] if tail else []
            buf_len = len(tail)
        buf.append(block_text)
        buf_len += block_len
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return [c for c in chunks if c.strip()]


def _is_table(block: list[str]) -> bool:
    return bool(block) and block[0].lstrip().startswith("|")


def chunk_document(
    meta: dict, body: str, chunk_size: int = 900, chunk_overlap: int = 120
) -> list[Chunk]:
    """Split one document body into :class:`Chunk` objects."""
    doc_id = str(meta.get("doc_id", "UNKNOWN"))
    lines = body.splitlines(keepends=True)

    # Identify section start offsets.
    starts: list[tuple[int, str]] = []  # (line index, heading)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m:
            starts.append((i, m.group(2).strip()))

    # Build (heading, text, span) triples.
    sections: list[tuple[str, str, list[int]]] = []
    if starts:
        # preamble = anything before the first "##"
        pre = "".join(lines[: starts[0][0]]).strip()
        if pre:
            sections.append(("", pre, [0, len("".join(lines[: starts[0][0]]))]))
        for idx, (line_idx, heading) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
            text = "".join(lines[line_idx:end]).strip()
            # drop the heading line itself from the citable text
            body_text = "".join(lines[line_idx + 1 : end]).strip()
            start_off = len("".join(lines[:line_idx]))
            end_off = len("".join(lines[:end]))
            sections.append((heading, body_text or text, [start_off, end_off]))
    else:
        sections.append(("", body.strip(), [0, len(body)]))

    chunks: list[Chunk] = []
    for heading, text, span in sections:
        code, title = "", heading
        m = _SECTION_CODE_RE.match(heading)
        if m:
            code, title = f"§{m.group(1)}", m.group(2).strip()
        for piece in _split_long_section(text, chunk_size, chunk_overlap):
            pid = hashlib.sha1(f"{doc_id}|{heading}|{piece}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{pid}",
                    doc_id=doc_id,
                    section=code,
                    section_title=title,
                    text=piece,
                    revision=str(meta.get("revision", "")),
                    effective_date=str(meta.get("effective_date", "")),
                    char_span=span,
                    doc_type=str(meta.get("doc_type", "")),
                    asset=str(meta.get("asset", "")),
                    title=str(meta.get("title", "")),
                    supersedes=meta.get("supersedes"),
                    supersedes_section=meta.get("supersedes_section"),
                )
            )
    return chunks


def build_chunks(
    corpus_dir: str | Path = "data/corpus", chunk_size: int = 900, chunk_overlap: int = 120
) -> list[Chunk]:
    """Chunk every ``*.md`` document in ``corpus_dir`` (sorted for determinism)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        meta, body = parse_document(path)
        chunks.extend(chunk_document(meta, body, chunk_size, chunk_overlap))
    return chunks


if __name__ == "__main__":  # pragma: no cover
    cs = build_chunks()
    print(f"{len(cs)} chunks from corpus")
