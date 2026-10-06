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
# Flattened / non-markdown coefficient tables
# --------------------------------------------------------------------------- #
# Many PDFs (esp. scanned journals via pdfplumber/OCR) never produce pipe
# tables — each cell becomes its own short paragraph. Without special
# handling, `_sliding_prose` then chops that run at prose_target_chars and
# the column headers land in one chunk while the coefficients land in the
# next, so detection can't tell what a number is an elasticity of. Detect
# these flattened coefficient blocks and treat them like real markdown
# tables: one atomic chunk with surrounding context.
_TABLE_CAPTION = re.compile(r"^\s*Table\s+\d+\b", re.I)
_COEF_LINE = re.compile(r"^-?\d+\.\d+\s*\*{0,3}\s*$")
_SE_OR_T_LINE = re.compile(r"^\(\s*-?\d+\.?\d*\s*\)\s*$")
_TABLE_FOOTER = re.compile(
    # Note:/Source: end with a colon (non-word char), so a trailing \b would
    # fail when the next char is a space — keep \b only on bare keywords.
    r"^\s*(Note\s*:|Source\s*:|System\s*R\s*[²2]\b|R-?squared\b|Observations\b)",
    re.I,
)
# Cap so a pathological run can't blow a single LLM context window; real
# regression tables are well under this.
_FLATTENED_TABLE_MAX_CHARS = 20000

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


def _line_table_signal(line: str) -> str:
    """Classify a single line for flattened-table detection.

    Returns one of: 'coef', 'stat', 'caption', 'footer', 'short', 'prose', 'blank'.
    """
    s = line.strip()
    if not s:
        return "blank"
    if _TABLE_CAPTION.match(s):
        return "caption"
    if _TABLE_FOOTER.match(s):
        return "footer"
    if _COEF_LINE.match(s) or (re.search(r"-?\d+\.\d+", s) and "*" in s and len(s) < 24):
        return "coef"
    if _SE_OR_T_LINE.match(s):
        return "stat"
    # Short label / header cells ("Cereals", "Fruits &", "Explanatory Variables")
    if len(s) <= 48 and not s.endswith("."):
        return "short"
    return "prose"


def _find_flattened_table_blocks(lines: List[str],
                                  occupied: List[Tuple[int, int]] | None = None
                                  ) -> List[Tuple[int, int]]:
    """Find spans that look like regression/result tables but weren't emitted
    as markdown pipes (OCR / pdfplumber layout flatten).

    A block starts at a ``Table N`` caption, or at a dense burst of coefficient
    / (se) lines. It then consumes short label lines, coefficients, t-stats,
    and the trailing Note/Source/R² footer, and stops at the next stretch of
    real prose (or the char cap). Spans overlapping ``occupied`` (already
    claimed by real markdown tables) are skipped.
    """
    occupied = occupied or []
    n = len(lines)
    signals = [_line_table_signal(ln) for ln in lines]

    # Boolean mask — O(1) occupied checks. The previous any()-over-spans
    # version was O(tables) per line and could dominate on long OCR docs.
    occupied_mask = [False] * n
    for a, b in occupied:
        lo = max(0, a)
        hi = min(n - 1, b)
        for idx in range(lo, hi + 1):
            occupied_mask[idx] = True

    def _is_occupied(idx: int) -> bool:
        return occupied_mask[idx]

    # Prefix counts of coef/stat lines so density queries are O(1).
    coefish = [0] * (n + 1)
    for i, s in enumerate(signals):
        coefish[i + 1] = coefish[i] + (1 if s in ("coef", "stat") else 0)

    def _coef_density(lo: int, hi: int) -> float:
        lo = max(0, lo)
        hi = min(n, hi)
        width = hi - lo
        if width <= 0:
            return 0.0
        return (coefish[hi] - coefish[lo]) / width

    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if _is_occupied(i):
            i += 1
            continue
        sig = signals[i]
        # Start: explicit caption, or a local burst of coef/stat lines.
        start = None
        if sig == "caption":
            start = i
        elif sig in ("coef", "stat") and _coef_density(i, min(n, i + 12)) >= 0.35:
            # Walk back over short labels / blanks that are likely column headers.
            start = i
            j = i - 1
            while j >= 0 and not _is_occupied(j) and signals[j] in ("short", "blank", "caption"):
                start = j
                if signals[j] == "caption":
                    break
                j -= 1
        if start is None:
            i += 1
            continue

        end = start
        chars = 0
        k = start
        while k < n and not _is_occupied(k):
            s = signals[k]
            line_len = len(lines[k]) + 1
            if chars + line_len > _FLATTENED_TABLE_MAX_CHARS and k > start:
                break
            if s == "prose":
                # Long Note:/Source: footers are still part of the table.
                if _TABLE_FOOTER.match(lines[k].strip()):
                    end = k
                    chars += line_len
                    k += 1
                    continue
                # Any other prose means the table body is over — do NOT absorb
                # the discussion sentence into the table chunk (PDF line-wrap
                # often leaves a mid-sentence fragment right after the grid).
                # Surrounding context is re-attached via paras_after_table.
                if k > start and _coef_density(start, k) >= 0.08:
                    break
                # Caption-only "Table N. …" followed by discussion, no coefs yet.
                if k - start > 3:
                    break
            end = k
            chars += line_len
            k += 1

        # Require a real coefficient signal — a lone "Table N" caption with
        # only prose after it is discussion of a table, not the table itself.
        if _coef_density(start, end + 1) >= 0.08 and (end - start) >= 4:
            blocks.append((start, end))
            i = end + 1
        else:
            i = start + 1

    # Merge overlapping / near-adjacent blocks. A caption block often abuts a
    # trailing coefficient run separated only by a Note:/blank gap (common
    # when the PDF puts overflow columns after the footnote). Do NOT merge
    # across a discussion/prose line — that used to glue hundreds of tables
    # into one giant chunk and stall corpus analyze.
    if not blocks:
        return []
    blocks.sort()
    merged: List[Tuple[int, int]] = [blocks[0]]
    for a, b in blocks[1:]:
        pa, pb = merged[-1]
        if a <= pb + 1:
            merged[-1] = (pa, max(pb, b))
            continue
        gap = lines[pb + 1:a]
        gap_ok = (
            a <= pb + 12
            and len(gap) > 0
            and all(_line_table_signal(ln) in ("blank", "footer", "short") for ln in gap)
        )
        if gap_ok:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


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
    md_table_blocks = _find_table_blocks(lines)
    # Flattened coefficient blocks (OCR / non-pipe tables) — skipped where a
    # real markdown table already claimed the lines.
    flat_blocks = _find_flattened_table_blocks(lines, occupied=md_table_blocks)
    # Prefer markdown tables when both fire on the same span; otherwise keep both.
    table_blocks: List[Tuple[int, int]] = list(md_table_blocks)
    for fb in flat_blocks:
        if not any(_spans_overlap(fb, mb) for mb in md_table_blocks):
            table_blocks.append(fb)
    table_blocks.sort()

    chunks: List[Chunk] = []
    consumed_spans: List[Tuple[int, int]] = []
    counter = 0

    # Only look ~N lines around each table for context paragraphs. Joining
    # lines[:start] / lines[end:] for every table is O(tables × doc) and can
    # freeze chunking on long OCR'd papers with many false-positive tables.
    _CTX_LINE_PAD = 80

    # ---- 1. table-anchored chunks (markdown pipes OR flattened coef blocks) ----
    for (start, end) in table_blocks:
        table_text = "\n".join(lines[start:end + 1])
        before_lo = max(0, start - _CTX_LINE_PAD)
        after_hi = min(len(lines), end + 1 + _CTX_LINE_PAD)
        before_text = "\n".join(lines[before_lo:start])
        after_text = "\n".join(lines[end + 1:after_hi])
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
    # Mask table lines so prose chunking doesn't re-include them. Build the
    # prose string from non-empty runs only so we don't materialize a second
    # full-size copy of a huge doc as mostly blank lines.
    if consumed_spans:
        occupied = [False] * len(lines)
        for (start, end) in consumed_spans:
            for idx in range(start, end + 1):
                if 0 <= idx < len(lines):
                    occupied[idx] = True
        prose_parts: List[str] = []
        gap = False
        for idx, ln in enumerate(lines):
            if occupied[idx]:
                gap = True
                continue
            if gap and prose_parts:
                prose_parts.append("")
                gap = False
            prose_parts.append(ln)
        prose = "\n".join(prose_parts)
    else:
        prose = "\n".join(lines)

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
