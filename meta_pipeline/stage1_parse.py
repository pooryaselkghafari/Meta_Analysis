"""Stage 1 — Parse PDFs into markdown and reconstruct split tables.

Primary path: Marker (https://github.com/datalab-to/marker), which converts PDFs
to markdown while preserving table structure. If Marker isn't installed, an
optional plaintext fallback keeps the rest of the pipeline runnable during
development.

Table reconstruction (joining tables split across pages) is stubbed with a clear
extension point — the heuristic is intentionally left simple until we validate on
real papers.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from .config import MarkerConfig
from .models import ParsedPaper


# --------------------------------------------------------------------------- #
# Marker integration
# --------------------------------------------------------------------------- #
def _marker_available() -> bool:
    try:
        import marker  # noqa: F401
        return True
    except Exception:
        return False


def _parse_with_marker(pdf_path: str, cfg: MarkerConfig) -> str:
    """Convert a single PDF to markdown using Marker's Python API.

    Marker's API has shifted across versions; this wraps the common entrypoint and
    is deliberately isolated so it can be swapped without touching the rest of the
    pipeline.
    """
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"force_ocr": cfg.force_ocr, "output_format": cfg.output_format},
    )
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    return text


def _parse_with_fallback(pdf_path: str) -> str:
    """Plaintext fallback (no table fidelity). Tries pdfminer, then pypdf."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        raise RuntimeError(f"No PDF backend available: {e}")


# --------------------------------------------------------------------------- #
# Table reconstruction (join tables split across pages)
# --------------------------------------------------------------------------- #
def reconstruct_split_tables(markdown: str) -> str:
    """Join markdown tables that were split across a page break.

    Heuristic placeholder: if a table block ends and, after only a page-number /
    whitespace gap, another markdown table begins with a matching column count,
    they are candidates for joining. For now this is a light pass; validate on real
    papers before making it aggressive (a wrong join corrupts data silently).

    TODO: strengthen using Marker's page metadata rather than text heuristics.
    """
    # Left intentionally conservative: return unchanged until validated.
    return markdown


# --------------------------------------------------------------------------- #
# Abstract extraction
# --------------------------------------------------------------------------- #
# Matches an "Abstract" heading on its own line (optionally markdown-styled:
# "# Abstract", "**Abstract**", "Abstract:").
_ABSTRACT_HEADING = re.compile(r"(?im)^\s*#{0,3}\s*\**\s*abstract\s*\**\s*:?\s*$")
# Section headings that mark the end of the abstract body.
_NEXT_SECTION_HEADING = re.compile(
    r"(?im)^\s*#{0,3}\s*\**\s*"
    r"(keywords?|jel\s*(classification|codes?)|"
    r"(?:1\.?|i\.?)\s*introduction|introduction)\b"
)


def _levenshtein(a: str, b: str) -> int:
    """Small dependency-free edit distance, used to fuzzy-match OCR-garbled
    section headings (e.g. scanned older working papers)."""
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


_ABSTRACT_WORD = "abstract"
# Only look for a fuzzy/OCR-garbled heading within roughly the first page or
# two — keeps the search cheap and avoids false positives from an unrelated
# short garbled token deeper in the paper.
_FUZZY_SEARCH_WINDOW = 6000


def _fuzzy_abstract_heading_end(markdown: str) -> Optional[int]:
    """Scan short, heading-like lines near the top of the paper for something
    that is 'abstract' after OCR noise: dropped/substituted letters
    ("Abstrct", "AB5TRACT", "bstract") or letter-spaced OCR output
    ("A B S T R A C T"). Returns the offset right after the matched line, or
    None if nothing close enough is found.
    """
    window = markdown[:_FUZZY_SEARCH_WINDOW]
    offset = 0
    for line in window.split("\n"):
        s = line.strip()
        line_len = len(line) + 1  # +1 for the newline consumed by split
        if not s or len(s) > 20:
            offset += line_len
            continue
        # Collapse letter-spaced OCR headings: "A B S T R A C T" -> "ABSTRACT"
        if re.fullmatch(r"(?:[A-Za-z]\s+){3,}[A-Za-z]", s):
            s = re.sub(r"\s+", "", s)
        candidate = re.sub(r"[^A-Za-z]", "", s).lower()
        # Word length should be in the neighborhood of "abstract" (8 chars) —
        # otherwise short unrelated headings ("data", "results") could match
        # within edit distance 2 by coincidence.
        if 5 <= len(candidate) <= 11:
            if _levenshtein(candidate, _ABSTRACT_WORD) <= 2:
                return offset + line_len
        offset += line_len
    return None


# Lines/paragraphs that look like title-block furniture rather than abstract
# prose — used by the positional fallback to skip past authors, affiliations,
# and copyright/received-date lines that sit between the title and the actual
# abstract text in older scanned working papers.
_TITLE_BLOCK_KEYWORDS = re.compile(
    r"(university|department|institute|email|e-mail|received|accepted|revised|"
    r"copyright|all rights reserved|corresponding author|selected paper|"
    r"working paper|discussion paper|issn|doi\s*:|orcid)", re.I
)


# Bound how far into the front matter we look for the *last* title/author/
# affiliation/copyright paragraph, so a short sentence deep inside the actual
# abstract body (e.g. a one-line closing sentence) can't be mistaken for front
# matter and push the start point too far forward.
_FRONT_MATTER_MAX_SCAN_CHARS = 2500
_FRONT_MATTER_MAX_SCAN_PARAS = 12
_KEYWORDS_OR_JEL = re.compile(r"(?i)\bkey\s*-?\s*words?\s*:|\bjel\b")


def _looks_like_front_matter(para: str) -> bool:
    if _TITLE_BLOCK_KEYWORDS.search(para):
        return True
    if len(para) < 40:
        return True
    if para.isupper():  # all-caps title line
        return True
    return False


def _positional_abstract_fallback(markdown: str, max_chars: int = 3000) -> Optional[str]:
    """Last-resort fallback for papers with no "Abstract" heading at all —
    common in older scanned/OCR'd working papers where the label was never
    captured as text (sometimes it was a distinct font/image the OCR dropped),
    or in preprints that simply don't label the abstract.

    Many such PDFs (especially double-spaced older papers) extract with one
    paragraph per physical line, so the abstract body itself is a run of many
    short paragraphs — a single "paragraph >= N chars" check misses it. Instead:
    find the last title/author/affiliation/copyright-looking paragraph within
    the early front matter, then accumulate every paragraph after that (up to
    a "Key words:"/"JEL" marker or the length cap) as the abstract.
    """
    window = markdown[:8000]
    end_m = _NEXT_SECTION_HEADING.search(window)
    search_region = window[:end_m.start()] if end_m else window
    paras = [p.strip() for p in re.split(r"\n\s*\n", search_region) if p.strip()]
    if not paras:
        return None

    scan_limit = min(len(paras), _FRONT_MATTER_MAX_SCAN_PARAS)
    chars_seen = 0
    last_front_matter_idx = -1
    for i in range(scan_limit):
        chars_seen += len(paras[i])
        if chars_seen > _FRONT_MATTER_MAX_SCAN_CHARS:
            break
        if _looks_like_front_matter(paras[i]):
            last_front_matter_idx = i

    collected: List[str] = []
    total = 0
    for p in paras[last_front_matter_idx + 1:]:
        if _KEYWORDS_OR_JEL.search(p) and collected:
            break
        collected.append(p)
        total += len(p)
        if total >= max_chars:
            break

    if not collected:
        return None
    abstract = " ".join(collected).strip()
    if len(abstract) < 100:
        return None
    return abstract[:max_chars]


def extract_abstract(markdown: str, max_chars: int = 3000) -> Optional[str]:
    """Best-effort extraction of the paper's abstract.

    This is deliberately separate from chunking/filtering: the abstract almost
    never contains a point estimate, so Stage 3's heuristic filter would drop it
    like any other estimate-free prose chunk. But it's the single cheapest piece
    of whole-paper context (what is this paper actually about, what's the DV/IV,
    what's the headline finding) to hand the Stage 4 main-extraction LLM alongside
    a table/text chunk it otherwise sees with no surrounding context. So it is
    captured once per paper here and carried on ParsedPaper.abstract rather than
    being subject to the estimate-signal filter at all.

    Four tiers, from strictest to most tolerant (older scanned papers run
    through OCR often garble the "Abstract" heading, or drop it entirely):
      1. Clean heading match ("# Abstract", "**Abstract**", "Abstract:").
      2. Fuzzy/OCR-tolerant heading match (dropped or substituted letters,
         letter-spaced headings) within the first ~page of text.
      3. Bare word "abstract" anywhere, as a near-last resort.
      4. Positional fallback: no heading found at all — take the first long,
         non-title-block paragraph before Introduction/Keywords.
    """
    m = _ABSTRACT_HEADING.search(markdown)
    start = m.end() if m else None
    if start is None:
        start = _fuzzy_abstract_heading_end(markdown)
    if start is None:
        m2 = re.search(r"(?i)\babstract\b\s*[:.\-]?\s*", markdown)
        if m2:
            start = m2.end()
    if start is None:
        return _positional_abstract_fallback(markdown, max_chars)
    window = markdown[start:start + max_chars * 2]
    end_m = _NEXT_SECTION_HEADING.search(window)
    end = end_m.start() if end_m else min(len(window), max_chars)
    abstract = window[:end].strip()
    # Guard against a false-positive heading match with no real body following.
    if len(abstract) < 40:
        return _positional_abstract_fallback(markdown, max_chars)
    return abstract


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def parse_paper(pdf_path: str, cfg: Optional[MarkerConfig] = None,
                paper_id: Optional[str] = None) -> ParsedPaper:
    cfg = cfg or MarkerConfig()
    paper_id = paper_id or Path(pdf_path).stem

    try:
        if _marker_available():
            md = _parse_with_marker(pdf_path, cfg)
        elif cfg.allow_fallback:
            md = _parse_with_fallback(pdf_path)
        else:
            raise RuntimeError("Marker not installed and fallback disabled.")
        md = reconstruct_split_tables(md)
        abstract = extract_abstract(md)
        return ParsedPaper(
            paper_id=paper_id, source_path=pdf_path, markdown=md, abstract=abstract,
        )
    except Exception as e:
        return ParsedPaper(
            paper_id=paper_id, source_path=pdf_path, markdown="",
            parse_ok=False, parse_error=str(e),
        )


def parse_corpus(input_dir: str, cfg: Optional[MarkerConfig] = None) -> List[ParsedPaper]:
    cfg = cfg or MarkerConfig()
    papers: List[ParsedPaper] = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(".pdf"):
            papers.append(parse_paper(os.path.join(input_dir, name), cfg))
    return papers
