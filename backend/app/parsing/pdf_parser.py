"""
PDF parser for academic papers using PyMuPDF.
Extracts: title, authors, abstract, year, DOI, section headings, page text with provenance.
All unknown fields are set to 'UNKNOWN', never invented.
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


@dataclass
class ParsedPage:
    page_number: int
    text: str
    section: str = ""


@dataclass
class ParsedPaper:
    document_hash: str
    title: str = "UNKNOWN"
    authors: str = "UNKNOWN"
    abstract: str = ""
    year: str = "UNKNOWN"
    doi: str = "UNKNOWN"
    study_design: str = "UNKNOWN"
    sample_size: str = "UNKNOWN"
    metadata_confidence: float = 0.0
    pages: list[ParsedPage] = field(default_factory=list)
    full_text: str = ""


SECTION_PATTERNS = [
    r'^abstract$',
    r'^introduction$',
    r'^background$',
    r'^methods?$',
    r'^materials?\s+and\s+methods?$',
    r'^results?$',
    r'^discussion$',
    r'^conclusion$',
    r'^references?$',
    r'^acknowledgements?$',
    r'^limitations?$',
    r'^related\s+work$',
    r'^experimental\s+setup$',
    r'^evaluation$',
]

STUDY_DESIGN_PATTERNS = [
    (r'meta[-\s]?analysis', 'Meta-analysis'),
    (r'systematic\s+review', 'Systematic Review'),
    (r'randomized\s+controlled\s+trial|RCT', 'Randomized Controlled Trial'),
    (r'cohort\s+study', 'Cohort Study'),
    (r'case[-\s]control', 'Case-Control Study'),
    (r'cross[-\s]sectional', 'Cross-Sectional Study'),
    (r'case\s+(study|report)', 'Case Study'),
    (r'pilot\s+study', 'Pilot Study'),
    (r'randomized|randomised', 'Randomized Controlled Trial'),
]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _is_section_heading(line: str) -> Optional[str]:
    stripped = line.strip().lower()
    for pat in SECTION_PATTERNS:
        if re.match(pat, stripped, re.IGNORECASE):
            return line.strip()
    # numbered section like '1. Introduction' or '2.1 Methods'
    m = re.match(r'^(\d+\.?\d*\.?)\s+([A-Z][\w\s]{2,40})$', line.strip())
    if m:
        return m.group(2).strip()
    return None


def _extract_year(text: str) -> str:
    years = re.findall(r'\b(19[6-9]\d|20[0-2]\d)\b', text)
    if years:
        # Return the most recent plausible year
        return max(years)
    return 'UNKNOWN'


def _extract_doi(text: str) -> str:
    m = re.search(r'\b(10\.\d{4,}/[^\s]+)', text)
    if m:
        doi = m.group(1).rstrip('.,')
        return doi
    return 'UNKNOWN'


def _extract_sample_size(text: str) -> str:
    # Look for 'n = 123' or 'N = 1,234' or 'n=400'
    patterns = [
        r'\bn\s*=\s*([0-9,]+)',
        r'N\s*=\s*([0-9,]+)',
        r'([0-9,]+)\s+participants',
        r'([0-9,]+)\s+subjects',
        r'([0-9,]+)\s+patients',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(',', '')
    return 'UNKNOWN'


def _extract_study_design(text: str) -> str:
    text_lower = text.lower()
    for pattern, label in STUDY_DESIGN_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return label
    return 'UNKNOWN'


def _extract_title_authors(first_page_text: str) -> tuple[str, str]:
    lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
    # Heuristic: title is often the longest bold-like line near the top
    # We use position heuristic: title = first substantial non-email, non-URL line
    title = 'UNKNOWN'
    authors = 'UNKNOWN'
    for i, line in enumerate(lines[:20]):
        if len(line) > 20 and not re.search(r'@|http|doi|vol\.|pp\.', line, re.I):
            title = line
            # Authors often follow immediately
            if i + 1 < len(lines):
                candidate = lines[i + 1]
                if re.search(r',\s*[A-Z]|\.\s+[A-Z]|and\s+[A-Z]', candidate):
                    authors = candidate
            break
    return title, authors


def _extract_abstract(full_text: str) -> str:
    m = re.search(
        r'abstract[:\s]+(.{100,2000})(?=\n\s*(?:introduction|background|keywords|1\s*\.?\s*introduction))',
        full_text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _clean_text(m.group(1))
    return ''


def parse_pdf(pdf_bytes: bytes) -> ParsedPaper:
    """Parse a PDF and return structured data. Works with or without PyMuPDF."""
    doc_hash = _hash_bytes(pdf_bytes)
    paper = ParsedPaper(document_hash=doc_hash)

    if not HAS_FITZ:
        paper.title = 'UNKNOWN (PyMuPDF not installed)'
        return paper

    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    except Exception as e:
        paper.title = f'UNKNOWN (parse error: {e})'
        return paper

    current_section = ''
    all_page_texts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text('text')
        lines = text.split('\n')

        # Detect section headings
        for line in lines:
            heading = _is_section_heading(line)
            if heading:
                current_section = heading

        cleaned = _clean_text(text)
        all_page_texts.append(cleaned)
        paper.pages.append(ParsedPage(
            page_number=page_num + 1,
            text=cleaned,
            section=current_section,
        ))

    doc.close()

    paper.full_text = ' '.join(all_page_texts)
    first_page = all_page_texts[0] if all_page_texts else ''
    first_pages = ' '.join(all_page_texts[:3])

    title, authors = _extract_title_authors(first_page)
    paper.title = title
    paper.authors = authors
    paper.abstract = _extract_abstract(first_pages)
    paper.year = _extract_year(first_pages)
    paper.doi = _extract_doi(paper.full_text)
    paper.sample_size = _extract_sample_size(paper.full_text)
    paper.study_design = _extract_study_design(paper.full_text)

    # Confidence: how many fields were found
    found = sum([
        paper.title != 'UNKNOWN',
        paper.authors != 'UNKNOWN',
        bool(paper.abstract),
        paper.year != 'UNKNOWN',
        paper.doi != 'UNKNOWN',
    ])
    paper.metadata_confidence = found / 5.0

    return paper


def parse_pdf_file(path: Path) -> ParsedPaper:
    """Convenience wrapper for file paths."""
    return parse_pdf(path.read_bytes())
