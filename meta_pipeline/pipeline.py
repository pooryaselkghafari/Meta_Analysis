"""Orchestrator — wires the stages together.

Flow (per the design doc):
  Stage 1  parse PDFs -> markdown (+ table reconstruction)
  Stage 3  table-aware chunking (deterministic)          [runs before Stage 2 on chunks]
  Stage 2  classify chunks by content type (cheap LLM)
  Stage 3  filter: heuristic signal + cheap-LLM detection
  Stage 4  merged extraction (main LLM) -> ExtractionRecord
  Stage 5  validation (LLM) + deterministic consistency checks
  (Stages 6-7 — estimate selection & meta-analysis — are the statistical layer,
   out of scope for this scaffolding.)

While prompts are blank, the LLM steps return None and the pipeline still runs,
emitting chunk-level output so the deterministic parts can be exercised.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional

from .config import PipelineConfig
from .models import (
    Chunk, ChunkType, ExtractionRecord, PaperMetadata, ParsedPaper,
    SourceLocation, TimePeriod, TransformationType,
)
from .llm import LLMClient
from . import stage1_parse, stage3_chunk


class Pipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.llm = LLMClient(self.cfg.model)
        self.targets = self._load_json(self.cfg.targets_path, default={})
        self.ontology = self._load_json(self.cfg.ontology_path, default={})

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(path: str, default):
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    # ------------------------------------------------------------------ #
    # Stage 1
    def parse(self) -> List[ParsedPaper]:
        return stage1_parse.parse_corpus(self.cfg.input_dir, self.cfg.marker)

    # Stage 3a — chunking (deterministic)
    def chunk(self, papers: List[ParsedPaper]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for paper in papers:
            chunks = stage3_chunk.chunk_paper(paper, self.cfg.chunk)
            paper.chunks = chunks
            all_chunks.extend(chunks)
        return all_chunks

    # Stage 2 — classification (cheap LLM; skipped if prompts blank)
    def classify(self, chunks: List[Chunk]) -> None:
        if not self.cfg.run_llm_stages:
            return
        for c in chunks:
            result = self.llm.classify_chunk(c.text)
            if not result:
                continue
            labels: List[ChunkType] = []
            for lbl in result.get("labels") or []:
                try:
                    labels.append(ChunkType(lbl))
                except ValueError:
                    continue
            c.labels = labels
            primary = result.get("primary")
            try:
                c.chunk_type = ChunkType(primary) if primary else (labels[0] if labels else ChunkType.OTHER)
            except ValueError:
                c.chunk_type = labels[0] if labels else ChunkType.OTHER
            c.classification_confidence = result.get("confidence")
            c.keep_classification = result.get("keep")

    # Stage 3b — filtering (heuristic + optional cheap-LLM detection)
    def filter_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        kept: List[Chunk] = []
        for c in chunks:
            # the abstract is exempt from the estimate-signal filter: it rarely
            # contains a point estimate but is valuable whole-paper context, so
            # it should never show up as "dropped" and should still reach
            # Stage 4 like any other kept chunk.
            if c.is_abstract:
                c.passes_filter = True
                c.filter_reason = "abstract_context"
                kept.append(c)
                continue
            # deterministic first pass (free)
            heuristic = stage3_chunk.heuristic_has_estimate(c, self.cfg.chunk)
            # the classifier's own keep/discard call is authoritative when
            # present (it already accounts for multi-label chunks, e.g. a
            # chunk that's both literature_review and result_text should be
            # kept) — only fall back to the old chunk_type-only heuristic
            # when classification hasn't run or didn't return a keep value.
            if c.keep_classification is False:
                c.passes_filter = False
                c.filter_reason = "classifier_discard"
                continue
            if c.keep_classification is None and c.chunk_type == ChunkType.LITERATURE_REVIEW:
                c.passes_filter = False
                c.filter_reason = "literature_review"
                continue
            if not heuristic:
                c.passes_filter = False
                c.filter_reason = "no_estimate_signal"
                continue
            # optional cheap-LLM confirmation
            if self.cfg.run_llm_stages:
                classification_context = {
                    "labels": [lbl.value for lbl in c.labels] if c.labels else (
                        [c.chunk_type.value] if c.chunk_type else []
                    ),
                    "primary": c.chunk_type.value if c.chunk_type else None,
                    "confidence": c.classification_confidence,
                    "keep": c.keep_classification,
                }
                result = self.llm.detect_estimate(c.text, self.targets, classification_context)
                if result:
                    c.detected = result.get("detected")
                    c.detection_confidence = result.get("confidence")
                    c.target_match = result.get("target_match")
                    c.keep_reason = result.get("keep_reason")
                if result and result.get("detected") is False:
                    c.passes_filter = False
                    c.filter_reason = "cheap_llm_negative"
                    continue
            c.passes_filter = True
            kept.append(c)
        return kept

    # Stage 4a — paper-level metadata (main LLM, once per paper; skipped if
    # prompts blank). Runs over the FULL chunk set (not just `kept`) — a
    # paragraph stating "450 households surveyed 2001-2015" rarely contains a
    # coefficient itself and would often be dropped by Stage 3's
    # estimate-signal filter, but it's exactly the text this pass needs.
    def extract_paper_metadata(self, papers: List[ParsedPaper],
                                chunks: List[Chunk]) -> Dict[str, PaperMetadata]:
        if not self.cfg.run_llm_stages:
            return {}
        by_paper: Dict[str, List[Chunk]] = {}
        for c in chunks:
            is_meta_chunk = c.chunk_type in (ChunkType.METHODOLOGY, ChunkType.DATA_DESCRIPTION) or any(
                lbl in (ChunkType.METHODOLOGY, ChunkType.DATA_DESCRIPTION) for lbl in (c.labels or [])
            )
            if is_meta_chunk:
                by_paper.setdefault(c.paper_id, []).append(c)
        abstract_by_paper = {p.paper_id: p.abstract for p in papers}
        result: Dict[str, PaperMetadata] = {}
        for paper_id, paper_chunks in by_paper.items():
            # cap length — this is context for a single summarizing call, not
            # the full paper; a handful of methodology/data paragraphs is
            # plenty and keeps the call cheap.
            text = "\n\n".join(c.text for c in paper_chunks)[:12000]
            raw = self.llm.extract_paper_metadata(abstract_by_paper.get(paper_id), text)
            if not raw:
                continue
            tp = raw.get("time_period") or {}
            result[paper_id] = PaperMetadata(
                paper_id=paper_id,
                model_type=raw.get("model_type"),
                countries_region=raw.get("countries_region"),
                time_period=TimePeriod(start=tp.get("start"), end=tp.get("end")),
                frequency=raw.get("frequency"),
                n_obs=raw.get("n_obs"),
                n_units=raw.get("n_units"),
                data_source=raw.get("data_source"),
            )
        return result

    @staticmethod
    def _backfill_paper_metadata(rec: ExtractionRecord, meta: Optional[PaperMetadata]) -> None:
        """Fill in paper-level fields from the once-per-paper Stage 4a pass,
        but only where this specific estimate's own chunk didn't already
        state something more specific — e.g. a robustness check run on a
        different subsample always keeps its own chunk-reported N/region
        rather than being overwritten by the paper-wide default."""
        if meta is None:
            return
        if rec.model_type is None:
            rec.model_type = meta.model_type
        if rec.countries_region is None:
            rec.countries_region = meta.countries_region
        if rec.time_period.start is None and rec.time_period.end is None:
            rec.time_period = meta.time_period
        if rec.frequency is None:
            rec.frequency = meta.frequency
        if rec.n_obs is None:
            rec.n_obs = meta.n_obs
        if rec.n_units is None:
            rec.n_units = meta.n_units
        if rec.data_source is None:
            rec.data_source = meta.data_source

    # Stage 4 — extraction (main LLM; skipped if prompts blank)
    def extract(self, chunks: List[Chunk],
                abstract_by_paper: Optional[Dict[str, str]] = None,
                paper_metadata: Optional[Dict[str, PaperMetadata]] = None) -> List[ExtractionRecord]:
        records: List[ExtractionRecord] = []
        if not self.cfg.run_llm_stages:
            return records
        abstract_by_paper = abstract_by_paper or {}
        paper_metadata = paper_metadata or {}
        for c in chunks:
            # Stages 2+3's own judgment for this chunk, handed to the main
            # model as prior-screening context (same pattern as Stage 3
            # receiving Stage 2's classification) — not re-derived from
            # scratch, but not blindly trusted either (see extraction_prompt).
            screening = {
                "labels": [lbl.value for lbl in c.labels] if c.labels else (
                    [c.chunk_type.value] if c.chunk_type else []
                ),
                "primary": c.chunk_type.value if c.chunk_type else None,
                "classification_confidence": c.classification_confidence,
                "keep_classification": c.keep_classification,
                "detected": c.detected,
                "detection_confidence": c.detection_confidence,
                "target_match": c.target_match,
                "keep_reason": c.keep_reason,
            }
            raw_estimates = self.llm.extract(
                c.text, self.targets, self.ontology,
                abstract=abstract_by_paper.get(c.paper_id),
                screening=screening,
            )
            for idx, raw in enumerate(raw_estimates or []):
                rec = self._record_from_raw(raw, c, idx)
                self._backfill_paper_metadata(rec, paper_metadata.get(c.paper_id))
                records.append(rec)
        return records

    # Stage 5 — validation (LLM + deterministic consistency check)
    def validate(self, records: List[ExtractionRecord],
                 chunk_by_id: Dict[str, Chunk]) -> None:
        for rec in records:
            # deterministic consistency (always runs)
            problems = rec.check_transformation_consistency()
            if problems:
                rec.requires_review = True
                rec.review_reason.extend(problems)
            # LLM validation (optional)
            if self.cfg.run_llm_stages:
                chunk = chunk_by_id.get(rec.source_chunk_id)
                if chunk is not None:
                    self.llm.validate(json.dumps(rec.to_dict()), chunk.text)
                    # merging validated fields back is left for prompt authoring

    # ------------------------------------------------------------------ #
    @staticmethod
    def _record_from_raw(raw: dict, chunk: Chunk, idx: int = 0) -> ExtractionRecord:
        """Map one raw LLM estimate dict onto an ExtractionRecord. `idx` is this
        estimate's position among however many the same chunk produced (a table
        can yield several), used to build a collision-free estimate_id since
        the model's own "estimate_label" is just a bookkeeping hint, not
        guaranteed unique or even present.

        Unknown keys ignored; enum coercion is defensive so a bad label
        doesn't crash the run. Provenance fields (source_location,
        table_complete, pages_used) are NOT trusted from the model — those are
        already known deterministically from the chunk itself (Stage 1/3
        parsing), so the model isn't asked for them except the finer-grained
        row/column identifiers a multi-row table needs, which get merged in
        below rather than overwriting the chunk-level fields."""
        rec = ExtractionRecord(
            paper_id=chunk.paper_id,
            estimate_id=f"{chunk.chunk_id}::est{idx}",
            source_chunk_id=chunk.chunk_id,
        )
        skip = {"paper_id", "estimate_id", "source_chunk_id",
                "source_location", "table_complete", "pages_used"}
        for key, val in raw.items():
            if hasattr(rec, key) and key not in skip:
                setattr(rec, key, val)
        # coerce transformation enum when present as a string
        for attr in ("elasticity_transformation_type",):
            v = getattr(rec, attr)
            if isinstance(v, str):
                try:
                    setattr(rec, attr, TransformationType(v))
                except ValueError:
                    setattr(rec, attr, TransformationType.UNKNOWN)
        # coerce a plain {"start":..,"end":..} dict into TimePeriod
        if isinstance(rec.time_period, dict):
            tp = rec.time_period
            rec.time_period = TimePeriod(start=tp.get("start"), end=tp.get("end"))
        # provenance always comes from the chunk, not the model — except
        # row/column, which the model is the only one positioned to know
        # (which row/column of a multi-estimate table this record came from)
        raw_row = raw.get("row") if isinstance(raw.get("row"), str) else None
        raw_col = raw.get("column") if isinstance(raw.get("column"), str) else None
        rec.source_location = SourceLocation(
            page=chunk.source_location.page,
            table=chunk.source_location.table,
            text_anchor=chunk.source_location.text_anchor,
            row=raw_row or chunk.source_location.row,
            column=raw_col or chunk.source_location.column,
        )
        rec.table_complete = chunk.table_complete
        rec.pages_used = list(chunk.pages_used)
        return rec

    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, object]:
        papers = self.parse()
        chunks = self.chunk(papers)
        self.classify(chunks)
        kept = self.filter_chunks(chunks)
        abstract_by_paper = {p.paper_id: p.abstract for p in papers if p.abstract}
        # Stage 4a runs over the full chunk set (before Stage 3 filtering),
        # not just `kept` — see extract_paper_metadata's docstring.
        paper_metadata = self.extract_paper_metadata(papers, chunks)
        records = self.extract(kept, abstract_by_paper, paper_metadata)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        self.validate(records, chunk_by_id)

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._write_outputs(papers, chunks, kept, records, paper_metadata)
        return {
            "papers": len(papers),
            "papers_failed": sum(1 for p in papers if not p.parse_ok),
            "chunks_total": len(chunks),
            "chunks_kept": len(kept),
            "records": len(records),
        }

    def _write_outputs(self, papers, chunks, kept, records, paper_metadata=None) -> None:
        out = self.cfg.output_dir
        with open(os.path.join(out, "chunks.json"), "w") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)
        with open(os.path.join(out, "chunks_kept.json"), "w") as f:
            json.dump([c.to_dict() for c in kept], f, indent=2)
        with open(os.path.join(out, "records.json"), "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        with open(os.path.join(out, "paper_metadata.json"), "w") as f:
            json.dump([m.to_dict() for m in (paper_metadata or {}).values()], f, indent=2)
        with open(os.path.join(out, "parse_report.json"), "w") as f:
            json.dump([
                {"paper_id": p.paper_id, "ok": p.parse_ok,
                 "error": p.parse_error, "n_chunks": len(p.chunks),
                 "abstract_found": bool(p.abstract)}
                for p in papers
            ], f, indent=2)
        with open(os.path.join(out, "abstracts.json"), "w") as f:
            json.dump([
                {"paper_id": p.paper_id, "abstract": p.abstract}
                for p in papers if p.abstract
            ], f, indent=2)
