"""
Semantic chunker for academic paper text.
Produces overlapping sliding-window chunks with section provenance.
"""
from dataclasses import dataclass
from typing import Optional

from app.parsing.pdf_parser import ParsedPage


@dataclass
class Chunk:
    page: int
    section: str
    text: str
    char_start: int = 0
    char_end: int = 0


def chunk_pages(
    pages: list[ParsedPage],
    chunk_size: int = 400,
    overlap: int = 80,
    min_chunk_len: int = 60,
) -> list[Chunk]:
    """
    Split pages into overlapping word-level chunks.
    Preserves section and page provenance for each chunk.
    """
    chunks: list[Chunk] = []

    for page in pages:
        text = page.text
        if not text or len(text) < min_chunk_len:
            continue

        words = text.split()
        pos = 0
        char_offset = 0

        while pos < len(words):
            window = words[pos: pos + chunk_size]
            chunk_text = ' '.join(window)
            if len(chunk_text) >= min_chunk_len:
                chunks.append(Chunk(
                    page=page.page_number,
                    section=page.section,
                    text=chunk_text,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_text),
                ))
            char_offset += len(' '.join(words[pos: pos + chunk_size - overlap])) + 1
            pos += chunk_size - overlap
            if pos >= len(words):
                break

    return chunks


def chunk_text(
    text: str,
    page: int = 1,
    section: str = '',
    chunk_size: int = 400,
    overlap: int = 80,
    min_chunk_len: int = 60,
) -> list[Chunk]:
    """Chunk a raw text string directly."""
    page_obj = ParsedPage(page_number=page, text=text, section=section)
    return chunk_pages([page_obj], chunk_size=chunk_size, overlap=overlap, min_chunk_len=min_chunk_len)
