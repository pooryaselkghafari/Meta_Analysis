# Macro Meta-Analysis Extraction Pipeline

Scaffolding for extracting point estimates from macroeconomics papers into a
structured dataset for meta-analysis. This implements the deterministic stages and
the **structure** of the LLM calls; the actual prompts are left blank on purpose.

## Status

| Part | State |
|---|---|
| Stage 1 — PDF parsing (Marker + pdfplumber/plaintext fallback) | Implemented |
| Table reconstruction (split tables) | Stub with extension point |
| Stage 3a — table-aware chunking | Implemented |
| Stage 3b — heuristic estimate filter | Implemented |
| Stage 2 — chunk classification (cheap LLM) | Call structure ready, **prompt blank** |
| Stage 3b — estimate detection (cheap LLM) | Call structure ready, **prompt blank** |
| Stage 4 — merged extraction (main LLM) | Call structure ready, **prompt blank** |
| Stage 5 — validation (LLM) + consistency check | Consistency check implemented; **LLM prompt blank** |
| Stages 6–7 — estimate selection & meta-analysis | Out of scope (statistical layer) |

## Layout

```
meta_pipeline/
  config.py        # all tunables: chunking, model names, paths
  models.py        # dataclasses = the consolidated schema (incl. transformation flags)
  stage1_parse.py  # Marker parsing + plaintext fallback + table-reconstruction hook
  stage3_chunk.py  # table-aware chunking + heuristic estimate filter
  prompts.py       # <-- BLANK prompt templates to fill in later
  llm.py           # two-tier LLM client (cheap + main); parses JSON; no-op if prompts blank
  pipeline.py      # orchestrator wiring the stages
run.py             # CLI
smoke_test.py      # deterministic test (no API, no Marker)
```

## Where the prompts go

All four prompts live in `prompts.py`, each a function returning `(system, user)`:

- `classification_prompt(chunk_text)` — Stage 2
- `detection_prompt(chunk_text, targets)` — Stage 3
- `extraction_prompt(chunk_text, targets, ontology)` — Stage 4 (the main one)
- `validation_prompt(record_json, chunk_text)` — Stage 5

While these return empty strings, `llm.py` detects the blank and returns `None`, so
the pipeline runs the deterministic stages and simply produces no extraction
records. Flip on LLM stages with `--llm` once prompts + `ANTHROPIC_API_KEY` are set.

## No-transformation policy

The pipeline **never converts** between functional forms. `models.py` records four
descriptive fields — `dv_is_raw`, `dv_transformation_type`, `iv_is_raw`,
`iv_transformation_type` — and `ExtractionRecord.check_transformation_consistency()`
enforces that `is_raw=True` pairs with `transformation_type="none"`. Contradictions
set `requires_review`.

## Run

```bash
pip install -r requirements.txt          # pdfplumber/pdfminer/pypdf fallback parsers + anthropic
# For real table fidelity also: pip install marker-pdf

# deterministic only (prompts still blank):
python run.py --input input_papers --output output

# with LLM stages (after prompts written):
export ANTHROPIC_API_KEY=...
python run.py --input input_papers --output output --llm

python smoke_test.py                      # quick sanity check
```

## Inputs to supply

- `targets.json` — user DV/IV targets
- `ontology.json` — accepted / related-separate / excluded mappings per target

Both are loaded if present and passed into the (blank) prompts; format is up to you
when you author the prompts.

## Dashboard (UI)

A local Flask dashboard covers the pipeline through the heuristic filter — more
pages will be added as the cheap-AI and main-AI stages come online.

```bash
pip install -r requirements.txt   # now includes flask + PyMuPDF
python webapp/app.py
# open http://127.0.0.1:5050
```

**Corpus page (`/`)** — drag-and-drop PDFs (saved to `input_papers/`), enter DV/IV
targets as tag inputs (saved to `targets.json`), then **Analyze** (or **Update
analysis** on repeat runs) — runs Stages 1–3 (deterministic; LLM stages stay off)
and shows a summary.

**Filter results page (`/results`)** — a thumbnail rail (first page of each PDF,
rendered via PyMuPDF, with a generic icon fallback if that's not installed) down
the left; clicking a paper shows its full chunk sequence in the main panel, styled
like manuscript markup: chunks that passed the heuristic filter appear normal with
a green rule, chunks that were dropped appear struck through and greyed with an
amber rule and the drop reason (e.g. `no_estimate_signal`). Table-bearing chunks
get a `table` tag. This is meant as an audit tool — if real tables are showing up
dropped, or obvious junk is passing, that's the heuristic in `stage3_chunk.py` to
tune before spending on LLM calls.

Known limitation: chunks currently display in creation order (all table-anchored
chunks, then all prose chunks), not document reading order, since `source_location`
page tracking isn't populated yet — see the open questions in the design doc.

## Notes

- Without Marker installed, Stage 1 falls back to `pdfplumber`, which detects tables
  geometrically (no ML models — much lighter than Marker) and still renders them as
  markdown tables, so they get the same table-aware chunking. Only if `pdfplumber`
  itself is unavailable does it drop further to plain pdfminer/pypdf text
  extraction, which loses table structure entirely. Install `marker-pdf` for the
  best fidelity if the server can afford the RAM/CPU (or GPU) it needs.
- Stage ordering follows the design doc: chunking is deterministic and runs before
  the cheap-LLM classification/detection so the expensive calls only see
  pre-filtered, table-aware chunks.
