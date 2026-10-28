"""
utils/chunker.py
Section-aware text splitter for Indian bare acts.

Strategy:
  1. Try to split on "Section N" boundaries (section-aware).
  2. If a section is still too long, sub-split with fixed size + overlap.
  3. Each chunk gets metadata: act, section, page, chunk_id.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Regex patterns for section headers in Indian law PDFs
SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Section|SEC\.|Sec\.?|SECTION)\s*(\d+[A-Z]?(?:\.\d+)?)\b",
    re.IGNORECASE,
)

PAGE_PATTERN = re.compile(r"\[Page (\d+)\]")


@dataclass
class Chunk:
    text: str
    act: str
    section: str = "Unknown"
    page: str = "Unknown"
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "act": self.act,
            "section": self.section,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


def chunk_document(
    text: str,
    act_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Split a full act text into Chunk objects.
    Returns a list of Chunk dataclass instances.
    """
    chunks: list[Chunk] = []

    # ── Detect page boundaries ──────────────────────────────────────────────
    # Replace [Page N] markers with a sentinel so we can track page numbers
    lines = text.split("\n")
    current_page = "1"
    processed_lines = []
    for line in lines:
        m = PAGE_PATTERN.search(line)
        if m:
            current_page = m.group(1)
            processed_lines.append(f"__PAGE__{current_page}__")
        else:
            processed_lines.append(line)
    text = "\n".join(processed_lines)

    # ── Split on Section headers ─────────────────────────────────────────────
    parts = SECTION_PATTERN.split(text)
    # parts alternates: [pre-section-text, section_num, section_body, ...]
    if len(parts) <= 1:
        # No section headers found — fall back to fixed-size splitting
        logger.debug(f"No section headers found in {act_name}, using fixed chunking.")
        return _fixed_chunks(text, act_name, chunk_size, overlap, current_page)

    # parts[0] = preamble, then pairs of (section_num, section_body)
    preamble = parts[0].strip()
    if preamble:
        chunks.extend(_fixed_chunks(preamble, act_name, chunk_size, overlap, "1", section="Preamble"))

    i = 1
    while i < len(parts) - 1:
        section_num = parts[i].strip()
        section_body = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract current page from body (last __PAGE__N__ marker)
        page_markers = re.findall(r"__PAGE__(\d+)__", section_body)
        section_page = page_markers[-1] if page_markers else current_page
        # Remove page markers from body
        section_body = re.sub(r"__PAGE__\d+__", "", section_body).strip()

        sub_chunks = _fixed_chunks(
            section_body,
            act_name,
            chunk_size,
            overlap,
            section_page,
            section=f"Section {section_num}",
        )
        chunks.extend(sub_chunks)
        i += 2

    # ── Assign chunk IDs ─────────────────────────────────────────────────────
    for idx, chunk in enumerate(chunks):
        chunk.chunk_id = f"{act_name.replace(' ', '_')}_{idx:04d}"

    logger.info(f"Chunked '{act_name}' → {len(chunks)} chunks.")
    return chunks


def _fixed_chunks(
    text: str,
    act_name: str,
    chunk_size: int,
    overlap: int,
    page: str,
    section: str = "Unknown",
) -> list[Chunk]:
    """Split text into fixed-size token-approximate chunks with overlap."""
    words = text.split()
    result = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        result.append(Chunk(
            text=chunk_text,
            act=act_name,
            section=section,
            page=page,
        ))
        if end == len(words):
            break
        start = end - overlap
    return result
