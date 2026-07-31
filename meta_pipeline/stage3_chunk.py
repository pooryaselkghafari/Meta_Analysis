"""Table-aware chunking.

Implements the design-doc chunking strategy: do NOT chunk by arbitrary character
count. When a markdown table is detected, the chunk is:

    [N paragraphs before] + [the entire table] + [N paragraphs after]

so that column headers (which define the specification), bottom-of-table notes
(N, F-stat, R-squared, fixed-effects rows) and the author's immediate
interpretation all stay attached to the coefficients.

Prose that is not near a table is chunked separately with light overlap, so that
estimates reported only in text (common in macro papers) are not lost.

This module is deterministic — no LLM. Classification (Stage 2) and the heuristic
estimate-signal filter (Stage 3) live here too, since they operate on the same
chunk objects.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .config import ChunkConfig
from .models import Chunk, ParsedPaper, SourceLocation


# A markdown table line looks like:  | a | b | c |
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")

# --------------------------------------------------------------------------- #
# Running header/footer boilerplate stripping
# --------------------------------------------------------------------------- #
# Journal page headers/footers get re-extracted on every page (e.g. "© Akdeniz
# University Faculty of Agriculture  Adekunle et al./Mediterr Agric Sci (2023)
# 36(1): 37-46"), which pollutes chunks with repeated noise unrelated to the
# paper's content. These lines are dropped before chunking so they never show
# up in table-anchored or prose chunks.
_COPYRIGHT_LINE = re.compile(r"©")
# "<Author> et al./<Journal Name> (<year>) <vol>(<issue>): <pages>" and similar
# citation-style running headers — the (year) vol(issue): pages tail is the
# distinctive, journal-agnostic signal.
_JOURNAL_CITATION_LINE = re.compile(
    r"\(\d{4}\)\s*\d+\s*\(\d+\)\s*:\s*\d+\s*[-–]\s*\d+"      # "(2023) 36(1): 37-46"
    r"|\d+\s*\(\d+\)\s*:\s*\d+\s*[-–]\s*\d+\s*,\s*\d{4}"      # "3(6): 705-720, 2014"
)
_RUNNING_HEADER_KEYWORDS = re.compile(
    r"all rights reserved|downloaded from|this content downloaded", re.I
)
# Running footer used by some journals (e.g. SCIENCEDOMAIN/AJAEES-style):
# "Sienso et al.; AJAEES, Article no. AJAEES.2014.6.021" — repeated on every
# page, and specific enough ("Article no.") not to collide with a normal
# in-text citation like "Smith et al. (2010) found...".
_ARTICLE_NO_LINE = re.compile(r"article\s*no\.?\s*[:\-]?\s*\S+", re.I)
# A line that is nothing but a page number.
_BARE_PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")


def _is_boilerplate_line(line: str) -> bool:
    """True if a line is running-header/footer noise, not paper content."""
    s = line.strip()
    if not s:
        return False
    if _BARE_PAGE_NUMBER_LINE.match(s):
        return True
    # Boilerplate lines are short running headers/footers, not body paragraphs;
    # cap length so a genuine long sentence that happens to mention "downloaded
    # from" or similar isn't dropped.
    if len(s) > 200:
        return False
    if _COPYRIGHT_LINE.search(s):
        return True
    if _JOURNAL_CITATION_LINE.search(s):
        return True
    if _RUNNING_HEADER_KEYWORDS.search(s):
        return True
    if _ARTICLE_NO_LINE.search(s):
        return True
    return False


def strip_boilerplate(markdown: str) -> str:
    """Remove running-header/footer lines (copyright notices, journal
    citation headers repeated on every page) from parsed markdown."""
    lines = markdown.split("\n")
    kept = [ln for ln in lines if not _is_boilerplate_line(ln)]
    return "\n".join(kept)


# --------------------------------------------------------------------------- #
# References/bibliography section stripping
# --------------------------------------------------------------------------- #
# A reference list is never a valid extraction target, but it isn't
# boilerplate in the line-by-line sense above — it's a genuine multi-paragraph
# section that happens to be full of numbers that look like point estimates
# to the heuristic filter (DOI codes like "10.2307/1913827" match the decimal
# regex, and citation titles routinely contain estimate keywords like "model"
# or "estimates"). Left in, it passes the free heuristic filter on false
# pretenses and gets sent to the paid cheap-AI classifier, which then
# correctly rejects it — but only after spending a call on it. Stripped here
# instead, before chunking, so it's never a candidate chunk at all.
_REFERENCES_HEADING = re.compile(
    r"^\s{0,3}[#*_]{0,3}\s*(references|bibliography|works\s+cited)\s*[#*_]{0,3}\s*:?\s*$", re.I
)
# Appendix content (including appendix tables) still counts as valid
# extraction material, so if an appendix section follows the references
# (a common layout), only the span between "References" and "Appendix" is
# removed — the appendix itself is left intact.
_APPENDIX_HEADING = re.compile(r"^\s{0,3}[#*_]{0,3}\s*appendix", re.I)


def strip_references_section(markdown: str) -> str:
    """Drop the references/bibliography section from parsed markdown, keeping
    any appendix that follows it intact. No-ops if no references heading is
    found (rather than guessing), so a paper whose reference list wasn't
    cleanly split out by parsing is left unchanged, same as before."""
    lines = markdown.split("\n")
    ref_start = None
    for i, line in enumerate(lines):
        if _REFERENCES_HEADING.match(line.strip()):
            ref_start = i
            break
    if ref_start is None:
        return markdown
    end = len(lines)
    for j in range(ref_start + 1, len(lines)):
        if _APPENDIX_HEADING.match(lines[j].strip()):
            end = j
            break
    return "\n".join(lines[:ref_start] + lines[end:])


def _split_paragraphs(text: str) -> List[str]:
    # Paragraphs separated by blank lines; keep non-empty.
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _find_table_blocks(lines: List[str]) -> List[Tuple[int, int]]:
    """Return (start_idx, end_idx) line spans for contiguous markdown tables."""
    blocks: List[Tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _TABLE_LINE.match(lines[i]):
            start = i
            while i < n and _TABLE_LINE.match(lines[i]):
                i += 1
            # a real table has at least a header + separator + one row
            if i - start >= 2:
                blocks.append((start, i - 1))
        else:
            i += 1
    return blocks


def _context_paragraphs(text_before: str, text_after: str,
                        n_before: int, n_after: int) -> Tuple[str, str]:
    before = _split_paragraphs(text_before)[-n_before:] if n_before else []
    after = _split_paragraphs(text_after)[:n_after] if n_after else []
    return "\n\n".join(before), "\n\n".join(after)


_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _chunk_contains_abstract(chunk_text: str, abstract: str) -> bool:
    """Match a chunk against the paper's abstract regardless of differences in
    line-wrapping/whitespace (the abstract was extracted from the same markdown
    but paragraph splitting can re-wrap it slightly)."""
    if not abstract:
        return False
    norm_chunk = _normalize(chunk_text)
    norm_abs = _normalize(abstract)
    # A short probe (first ~80 chars) is enough to identify the match and is
    # robust to the abstract being truncated (max_chars) relative to the chunk.
    probe = norm_abs[:80]
    return bool(probe) and probe in norm_chunk


def chunk_paper(paper: ParsedPaper, cfg: ChunkConfig) -> List[Chunk]:
    """Produce table-aware chunks plus prose chunks for one parsed paper."""
    if not paper.parse_ok or not paper.markdown.strip():
        return []

    md = strip_boilerplate(paper.markdown)
    md = strip_references_section(md)
    lines = md.split("\n")
    table_blocks = _find_table_blocks(lines)

    chunks: List[Chunk] = []
    consumed_spans: List[Tuple[int, int]] = []
    counter = 0

    # ---- 1. table-anchored chunks ----
    for (start, end) in table_blocks:
        table_text = "\n".join(lines[start:end + 1])
        before_text = "\n".join(lines[:start])
        after_text = "\n".join(lines[end + 1:])
        before, after = _context_paragraphs(
            before_text, after_text, cfg.paras_before_table, cfg.paras_after_table
        )
        combined = "\n\n".join(x for x in [before, table_text, after] if x)
        counter += 1
        chunks.append(Chunk(
            chunk_id=f"{paper.paper_id}::c{counter:03d}",
            paper_id=paper.paper_id,
            text=combined,
            contains_table=True,
            source_location=SourceLocation(),
        ))
        consumed_spans.append((start, end))

    # ---- 2. prose chunks from the remaining (non-table) text ----
    # Mask table lines so prose chunking doesn't re-include them.
    masked_lines = list(lines)
    for (start, end) in consumed_spans:
        for idx in range(start, end + 1):
            masked_lines[idx] = ""
    prose = "\n".join(masked_lines)
    for para_block in _sliding_prose(prose, cfg):
        counter += 1
        chunks.append(Chunk(
            chunk_id=f"{paper.paper_id}::c{counter:03d}",
            paper_id=paper.paper_id,
            text=para_block,
            contains_table=False,
            is_abstract=_chunk_contains_abstract(para_block, paper.abstract or ""),
        ))

    return chunks


def _sliding_prose(text: str, cfg: ChunkConfig) -> List[str]:
    """Chunk prose into ~prose_target_chars windows with overlap, respecting
    paragraph boundaries where possible."""
    paras = _split_paragraphs(text)
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for para in paras:
        if size + len(para) > cfg.prose_target_chars and buf:
            chunks.append("\n\n".join(buf))
            # start next buffer with tail overlap
            overlap, o = [], 0
            for p in reversed(buf):
                if o + len(p) > cfg.prose_overlap_chars:
                    break
                overlap.insert(0, p)
                o += len(p)
            buf = list(overlap)
            size = sum(len(p) for p in buf)
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# --------------------------------------------------------------------------- #
# Heuristic estimate-signal filter (deterministic part of Stage 3)
# --------------------------------------------------------------------------- #
_SE_IN_PARENS = re.compile(r"\(\s*\d+\.\d+\s*\)")          # e.g. (0.12)
_STARS = re.compile(r"\*{1,3}")                            # significance stars
_NUM = re.compile(r"-?\d+\.\d+")


def heuristic_has_estimate(chunk: Chunk, cfg: ChunkConfig) -> bool:
    """Cheap, deterministic signal that a chunk may contain a point estimate.
    Used to pre-filter before the cheap-LLM detection call, cutting token cost."""
    text = chunk.text.lower()
    if _SE_IN_PARENS.search(chunk.text):
        return True
    if _STARS.search(chunk.text) and _NUM.search(chunk.text):
        return True
    if any(kw in text for kw in cfg.estimate_keywords) and _NUM.search(chunk.text):
        return True
    # tables are always worth a closer look
    return chunk.contains_table
