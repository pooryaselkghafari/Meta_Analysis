"""Flask dashboard for the meta-analysis extraction pipeline.

Pages, matching how far the pipeline currently runs:

  /             upload page — drag-drop PDFs, enter DV/IV targets, run analysis
  /results      filter results — thumbnail rail + kept/dropped chunk markup per paper
  /cheap-ai     Stage 2 (cheap AI #1) — classify kept chunks by content type
  /detection-ai Stage 3 (cheap AI #2) — screen kept chunks for a point-estimate signal
  /main-ai      Stage 4 (main/expensive AI #3) — extract structured point-estimate
                records from every chunk detection said yes to; shown as a table
  /settings     API key + model configuration for the three AI slots
                (cheap / main / validation), plus the five prompt templates

The Corpus and Filter results pages run the deterministic stages only
(Stages 1-3's heuristic filter) and never call an LLM. Cheap AI, Detection AI,
and Main AI are the pages that call a model, using whichever settings are
saved on the Settings page. All three refuse to run — returning a clear 400
error instead of silently doing nothing — if the relevant AI's API key isn't
set or its prompt is still blank; see _ai_readiness_error below. Extraction
results live in <project>/output/records.json — one row per extracted point
estimate, rebuilt (per-chunk) every time Main AI is run or re-run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, abort
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meta_pipeline import Pipeline, PipelineConfig, LLMClient, settings_store  # noqa: E402
from meta_pipeline import prompts_store, AVAILABLE_MODELS, effort_levels_for  # noqa: E402
from meta_pipeline.models import ChunkType  # noqa: E402
import regression  # noqa: E402 — Stage 5: meta-regression over records.json

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Projects — each analysis run (its own papers, targets, AI settings, prompts,
# and output) lives in its own projects/<project_id>/ directory instead of
# one global corpus at the repo root. This is what lets you run a second,
# unrelated paper set without it overwriting or mixing with the first, and is
# also the seam a future multi-user version would hang a user_id off of.
#
# INPUT_DIR/OUTPUT_DIR/THUMB_DIR/TARGETS_PATH/ONTOLOGY_PATH below are
# reassigned by _apply_active_project() whenever the active project changes
# (on startup, and whenever the user switches projects from the UI). Every
# route reads them as plain module globals at call time, so switching
# projects redirects all file I/O immediately without touching each route.
# --------------------------------------------------------------------------- #
PROJECTS_DIR = BASE_DIR / "projects"
ACTIVE_PROJECT_FILE = BASE_DIR / "active_project.json"

INPUT_DIR = OUTPUT_DIR = THUMB_DIR = TARGETS_PATH = ONTOLOGY_PATH = None  # set below


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "project"


def _unique_project_id(name: str) -> str:
    base = _slugify(name)
    candidate = base
    i = 2
    while (PROJECTS_DIR / candidate).exists():
        candidate = f"{base}-{i}"
        i += 1
    return candidate


def _project_meta_path(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "project.json"


def _read_project_meta(project_id: str) -> dict:
    path = _project_meta_path(project_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"id": project_id, "name": project_id}


def _list_projects() -> list:
    if not PROJECTS_DIR.exists():
        return []
    out = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        if d.is_dir():
            meta = _read_project_meta(d.name)
            out.append({"id": d.name, "name": meta.get("name", d.name)})
    return out


def _create_project(name: str) -> str:
    project_id = _unique_project_id(name or "project")
    proj_dir = PROJECTS_DIR / project_id
    (proj_dir / "input_papers").mkdir(parents=True, exist_ok=True)
    (proj_dir / "output" / "thumbnails").mkdir(parents=True, exist_ok=True)
    _project_meta_path(project_id).write_text(
        json.dumps({"id": project_id, "name": name or project_id}, indent=2)
    )
    return project_id


def _get_active_project_id() -> str:
    if ACTIVE_PROJECT_FILE.exists():
        try:
            pid = json.loads(ACTIVE_PROJECT_FILE.read_text()).get("project_id")
            if pid and (PROJECTS_DIR / pid).exists():
                return pid
        except (json.JSONDecodeError, OSError):
            pass
    projects = _list_projects()
    if projects:
        return projects[0]["id"]
    return _create_project("Default")  # nothing exists yet on a fresh install


def _set_active_project_id(project_id: str) -> None:
    ACTIVE_PROJECT_FILE.write_text(json.dumps({"project_id": project_id}, indent=2))


def _migrate_legacy_root_data() -> None:
    """One-time migration for installs that predate the project system: if
    projects/ doesn't exist yet but old root-level data does (input_papers/,
    output/, ai_settings.json, etc. sitting directly at the repo root), move
    it into projects/default/ instead of losing it, and make "default" the
    active project. No-ops (does nothing further) once projects/ exists."""
    if PROJECTS_DIR.exists():
        return
    legacy_paths = [
        BASE_DIR / "input_papers", BASE_DIR / "output",
        BASE_DIR / "ai_settings.json", BASE_DIR / "prompts_settings.json",
        BASE_DIR / "targets.json", BASE_DIR / "ontology.json",
    ]
    has_legacy_data = any(p.exists() for p in legacy_paths)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not has_legacy_data:
        return
    default_dir = PROJECTS_DIR / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for p in legacy_paths:
        if p.exists():
            shutil.move(str(p), str(default_dir / p.name))
    _project_meta_path("default").write_text(json.dumps({"id": "default", "name": "Default"}, indent=2))
    _set_active_project_id("default")


def _apply_active_project() -> None:
    """Recompute every path global from whichever project is currently
    active, and repoint settings_store/prompts_store at the same directory.
    Called on startup and immediately after creating/switching projects."""
    global INPUT_DIR, OUTPUT_DIR, THUMB_DIR, TARGETS_PATH, ONTOLOGY_PATH
    project_id = _get_active_project_id()
    proj_dir = PROJECTS_DIR / project_id
    INPUT_DIR = proj_dir / "input_papers"
    OUTPUT_DIR = proj_dir / "output"
    THUMB_DIR = OUTPUT_DIR / "thumbnails"
    TARGETS_PATH = proj_dir / "targets.json"
    ONTOLOGY_PATH = proj_dir / "ontology.json"
    for d in (INPUT_DIR, OUTPUT_DIR, THUMB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    settings_store.set_base_dir(proj_dir)
    prompts_store.set_base_dir(proj_dir)


_migrate_legacy_root_data()
_apply_active_project()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB total upload cap


@app.route("/api/projects", methods=["GET"])
def api_list_projects():
    return jsonify({"projects": _list_projects(), "active_project_id": _get_active_project_id()})


@app.route("/api/projects", methods=["POST"])
def api_create_project():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "project name is required"}), 400
    project_id = _create_project(name)
    _set_active_project_id(project_id)
    _apply_active_project()
    return jsonify({"project_id": project_id, "name": name})


@app.route("/api/projects/<project_id>/activate", methods=["POST"])
def api_activate_project(project_id):
    if not (PROJECTS_DIR / project_id).exists():
        return jsonify({"error": "project not found"}), 404
    _set_active_project_id(project_id)
    _apply_active_project()
    return jsonify({"project_id": project_id})


# --------------------------------------------------------------------------- #
# Pipeline stage state — tracks how far the user has actually progressed, so
# a page whose input hasn't been produced yet can refuse to open (server
# side) and its nav link can grey out (client side).
#
# Deliberately NOT a separately-persisted flag file: an earlier version of
# this tracked done/not-done in its own pipeline_state.json, which went stale
# the moment the underlying output was deleted or regenerated some other way
# (e.g. output files removed by hand, or a run interrupted) — the flag file
# would still say "done" even though the data behind it was gone, so pages
# stayed unlocked when they shouldn't have. Instead, "done" is recomputed
# from the actual output files on every request: it can never drift out of
# sync with reality because there's nothing to fall out of sync.
# --------------------------------------------------------------------------- #
STAGE_ORDER = ["corpus", "cheap_ai", "detection_ai", "main_ai", "regression", "dashboard"]
# Human label + the page where each stage is actually run — used both for
# "you need X first" messaging on a locked page and for the link that gets
# you there.
STAGE_LABELS = {
    "corpus": "Corpus analysis",
    "cheap_ai": "Cheap AI — Classification",
    "detection_ai": "Detection AI — Detection",
    "main_ai": "Main AI — Extraction",
    "regression": "Regression — Meta-analysis",
    "dashboard": "Dashboard — Key findings",
}
STAGE_HREFS = {
    "corpus": "/", "cheap_ai": "/cheap-ai", "detection_ai": "/detection-ai",
    "main_ai": "/main-ai", "regression": "/regression", "dashboard": "/dashboard",
}
# Which stage must be *_done before a given page is allowed to open at all.
# "/" (Corpus/upload) and "/settings" have no prerequisite. Regression and
# Dashboard both gate on main_ai_done for consistency with every other
# stage's strict gating, even though technically only *some* extraction
# records would suffice — the user must finish Stage 4 extraction fully
# before either unlocks. Dashboard doesn't require any regression runs to
# exist (it just shows an empty state for that section if none do yet),
# since summary stats over the extracted records are useful on their own.
PAGE_REQUIRES = {
    "results": "corpus", "cheap_ai": "corpus", "detection_ai": "cheap_ai",
    "main_ai": "detection_ai", "regression": "main_ai", "dashboard": "main_ai",
}


def _extractable_chunks(chunks: list) -> list:
    """Chunks eligible for Stage 4 extraction: classifiable (not abstract,
    not heuristic/cheap-LLM dropped) AND Stage 3 detection actually said yes.
    A chunk detection said no to is already excluded via passes_filter, and
    one detection hasn't screened yet (detected is None) shouldn't normally
    exist once detection_ai_done is true — excluded here anyway, defensively."""
    return [c for c in _classifiable_chunks(chunks) if c.get("detected") is True]


def _load_state() -> dict:
    """Derive each stage's done/not-done status directly from what's actually
    on disk right now, rather than trusting a separately-tracked flag."""
    corpus_done = (OUTPUT_DIR / "chunks.json").exists() and (OUTPUT_DIR / "chunks_kept.json").exists()

    classifiable = _classifiable_chunks(_kept_chunks()) if corpus_done else []
    cheap_ai_done = corpus_done and bool(classifiable) and all(c.get("chunk_type") for c in classifiable)

    detection_ai_done = cheap_ai_done and bool(classifiable) and all(
        c.get("detected") is not None for c in classifiable
    )

    extractable = _extractable_chunks(classifiable) if detection_ai_done else []
    main_ai_done = detection_ai_done and bool(extractable) and all(
        c.get("extraction_done") for c in extractable
    )

    return {
        "corpus_done": corpus_done,
        "cheap_ai_done": cheap_ai_done,
        "detection_ai_done": detection_ai_done,
        "main_ai_done": main_ai_done,
    }


def _discard_extraction_results() -> None:
    """Undo Stage 4 extraction's effect — clears 'extraction_done' on every
    chunk and empties records.json — since Stage 3 detection just (re-)ran,
    either directly or as a consequence of classification re-running, and any
    existing extraction records were built from a now-stale detected/kept
    set. Mirrors _discard_detection_results' reasoning one stage further
    downstream."""
    chunks = _kept_chunks()
    changed = False
    for c in chunks:
        if c.get("extraction_done") is not None:
            c["extraction_done"] = None
            changed = True
    if changed:
        _save_kept_chunks(chunks)
    (OUTPUT_DIR / "records.json").write_text(json.dumps([], indent=2))


def _discard_detection_results() -> None:
    """Undo Stage 3 detection's effect on chunks_kept.json — clears the
    'detected' flag and reverts any chunk that detection dropped back to
    passing the filter — since classification just re-ran and detection's
    verdicts are based on a (possibly now-stale) prior classification pass.
    This is also what makes detection_ai_done correctly recompute to False
    right after a fresh classify run, since _load_state derives it from these
    same 'detected' fields. Extraction is even further downstream than
    detection, so it's invalidated too — see _discard_extraction_results."""
    chunks = _kept_chunks()
    changed = False
    for c in chunks:
        if c.get("detected") is not None:
            c["detected"] = None
            changed = True
        if c.get("filter_reason") == "cheap_llm_negative":
            c["passes_filter"] = True
            c["filter_reason"] = None
            changed = True
    if changed:
        _save_kept_chunks(chunks)
    _discard_extraction_results()


@app.route("/api/pipeline/state", methods=["GET"])
def pipeline_state():
    return jsonify(_load_state())


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def _locked_response(page_key: str, page_title: str):
    """If page_key's prerequisite stage hasn't completed, render a locked
    placeholder instead of the real page — covers direct navigation/bookmarks,
    not just the greyed-out nav link."""
    required = PAGE_REQUIRES.get(page_key)
    if not required:
        return None
    state = _load_state()
    if state[f"{required}_done"]:
        return None
    return render_template(
        "locked.html",
        page_title=page_title,
        required_label=STAGE_LABELS[required],
        required_href=STAGE_HREFS[required],
    )


@app.route("/")
def upload_page():
    return render_template("upload.html")


@app.route("/results")
def results_page():
    locked = _locked_response("results", "Filter results")
    return locked or render_template("results.html")


@app.route("/cheap-ai")
def cheap_ai_page():
    locked = _locked_response("cheap_ai", "Cheap AI")
    return locked or render_template("cheap_ai.html")


@app.route("/detection-ai")
def detection_ai_page():
    locked = _locked_response("detection_ai", "Detection AI")
    return locked or render_template("detection_ai.html")


@app.route("/main-ai")
def main_ai_page():
    locked = _locked_response("main_ai", "Main AI")
    return locked or render_template("main_ai.html")


@app.route("/regression")
def regression_page():
    locked = _locked_response("regression", "Regression")
    return locked or render_template("regression.html")


@app.route("/dashboard")
def dashboard_page():
    locked = _locked_response("dashboard", "Dashboard")
    return locked or render_template("dashboard.html")


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


# --------------------------------------------------------------------------- #
# Papers: upload / list / delete
# --------------------------------------------------------------------------- #
def _paper_id_for(filename: str) -> str:
    return Path(secure_filename(filename)).stem


@app.route("/api/papers", methods=["GET"])
def list_papers():
    papers = []
    for f in sorted(INPUT_DIR.glob("*.pdf")):
        papers.append({
            "paper_id": f.stem,
            "filename": f.name,
            "size_bytes": f.stat().st_size,
        })
    return jsonify({"papers": papers, "has_results": (OUTPUT_DIR / "chunks.json").exists()})


@app.route("/api/papers/upload", methods=["POST"])
def upload_papers():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files received"}), 400
    saved, rejected = [], []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            rejected.append({"filename": f.filename, "reason": "not a PDF"})
            continue
        safe = secure_filename(f.filename)
        f.save(INPUT_DIR / safe)
        saved.append(safe)
    return jsonify({"saved": saved, "rejected": rejected})


@app.route("/api/papers/<paper_id>", methods=["DELETE"])
def delete_paper(paper_id):
    matches = list(INPUT_DIR.glob(f"{paper_id}.pdf"))
    if not matches:
        return jsonify({"error": "not found"}), 404
    matches[0].unlink()
    return jsonify({"deleted": paper_id})


# --------------------------------------------------------------------------- #
# Targets — elasticity types (e.g. "Own-price Elasticity") and the products
# they're calculated on (e.g. "maize", "wheat"). Cross-price elasticities
# match a pair of products drawn from this same product list — see
# meta_pipeline.prompts' extraction prompt for how that combination is
# resolved; nothing about the target list itself needs to know about pairing.
# --------------------------------------------------------------------------- #
@app.route("/api/targets", methods=["GET"])
def get_targets():
    if TARGETS_PATH.exists():
        return jsonify(json.loads(TARGETS_PATH.read_text()))
    return jsonify({"elasticities": [], "products": []})


@app.route("/api/targets", methods=["POST"])
def save_targets():
    data = request.get_json(force=True)
    elasticities = [e.strip() for e in data.get("elasticities", []) if e.strip()]
    products = [p.strip() for p in data.get("products", []) if p.strip()]
    TARGETS_PATH.write_text(json.dumps({"elasticities": elasticities, "products": products}, indent=2))
    return jsonify({"elasticities": elasticities, "products": products})


@app.route("/api/food-groups", methods=["GET"])
def get_food_groups():
    """The fixed standard 8-group food classification (see FoodGroup in
    meta_pipeline/models.py) — unlike elasticities/products, this isn't
    user-editable per project, just a reference vocab the Dashboard's filter
    bar and the Main AI edit form need."""
    return jsonify({"food_groups": list(LLMClient._FOOD_GROUPS)})


# --------------------------------------------------------------------------- #
# Analysis (deterministic stages only, for now)
# --------------------------------------------------------------------------- #
@app.route("/api/analyze", methods=["POST"])
def analyze():
    n_papers = len(list(INPUT_DIR.glob("*.pdf")))
    if n_papers == 0:
        return jsonify({"error": "no papers uploaded"}), 400

    cfg = PipelineConfig(
        input_dir=str(INPUT_DIR),
        output_dir=str(OUTPUT_DIR),
        targets_path=str(TARGETS_PATH),
        ontology_path=str(ONTOLOGY_PATH),
        run_llm_stages=False,  # heuristic filter only — cheap-AI stages come later
    )
    # Always build from the Settings page's saved models/keys, even though this
    # endpoint doesn't call the LLM yet (run_llm_stages=False) — so the moment
    # LLM stages are turned on here, they immediately honor whatever is
    # configured on /settings rather than silently using the code defaults.
    cfg.model = settings_store.load_model_config()
    pipeline = Pipeline(cfg)
    summary = pipeline.run()
    _clear_thumbnail_cache_for_missing_papers()
    # No explicit "mark done" needed — pipeline.run() just rewrote
    # chunks_kept.json from scratch with fresh, unclassified Chunk objects, so
    # _load_state() will naturally recompute cheap_ai_done/detection_ai_done
    # as False on the next read, re-greying those pages automatically.
    return jsonify(summary)


# --------------------------------------------------------------------------- #
# Results: per-paper chunk markup (kept vs. dropped by the heuristic filter)
# --------------------------------------------------------------------------- #
def _load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


@app.route("/api/results/papers", methods=["GET"])
def results_papers():
    report = _load_json(OUTPUT_DIR / "parse_report.json", [])
    chunks = _load_json(OUTPUT_DIR / "chunks.json", [])
    by_paper = {}
    for c in chunks:
        by_paper.setdefault(c["paper_id"], {"total": 0, "kept": 0})
        by_paper[c["paper_id"]]["total"] += 1
        if c.get("passes_filter"):
            by_paper[c["paper_id"]]["kept"] += 1

    out = []
    for p in report:
        stats = by_paper.get(p["paper_id"], {"total": 0, "kept": 0})
        out.append({
            "paper_id": p["paper_id"],
            "ok": p["ok"],
            "error": p.get("error"),
            "total_chunks": stats["total"],
            "kept_chunks": stats["kept"],
        })
    return jsonify({"papers": out})


@app.route("/api/results/papers/<paper_id>/chunks", methods=["GET"])
def results_paper_chunks(paper_id):
    chunks = _load_json(OUTPUT_DIR / "chunks.json", [])
    paper_chunks = [c for c in chunks if c["paper_id"] == paper_id]
    if not paper_chunks:
        return jsonify({"paper_id": paper_id, "chunks": []})
    return jsonify({"paper_id": paper_id, "chunks": paper_chunks})


# --------------------------------------------------------------------------- #
# Manual overrides — a human can correct any stage's automatic call rather
# than only being able to accept or fully re-run it. Shared helpers below are
# used by all four stages' override endpoints.
# --------------------------------------------------------------------------- #
def _find_chunk(chunks: list, chunk_id: str):
    return next((c for c in chunks if c.get("chunk_id") == chunk_id), None)


def _purge_records_for_chunk(chunk_id: str) -> None:
    """Drop any extraction records sourced from this chunk — used whenever a
    human edit at an earlier stage (drop/return, relabel, re-screen) makes an
    existing record's provenance stale. Scoped to just this one chunk rather
    than the blanket _discard_extraction_results(), since a single manual
    correction shouldn't throw away every other chunk's already-correct
    extraction work."""
    records = _load_json(OUTPUT_DIR / "records.json", [])
    filtered = [r for r in records if r.get("source_chunk_id") != chunk_id]
    if len(filtered) != len(records):
        (OUTPUT_DIR / "records.json").write_text(json.dumps(filtered, indent=2))


@app.route("/api/results/papers/<paper_id>/chunks/<path:chunk_id>/override", methods=["POST"])
def results_chunk_override(paper_id, chunk_id):
    """Stage 1 manual edit: drop a chunk the heuristic filter kept, or return
    one it dropped. Keeps chunks.json (the full per-paper display source) and
    chunks_kept.json (what downstream stages actually iterate over) in sync:
    returning a chunk adds it back to the kept set in a fresh, unclassified
    state (it needs to go through Cheap AI / Detection AI again, same as any
    newly-kept chunk); dropping one removes it from the kept set entirely and
    purges any extraction records already built from it."""
    data = request.get_json(force=True) or {}
    passes_filter = data.get("passes_filter")
    if not isinstance(passes_filter, bool):
        return jsonify({"error": "passes_filter (true/false) is required"}), 400

    all_chunks = _load_json(OUTPUT_DIR / "chunks.json", [])
    chunk = _find_chunk(all_chunks, chunk_id)
    if chunk is None:
        return jsonify({"error": "chunk not found"}), 404

    chunk["passes_filter"] = passes_filter
    chunk["filter_reason"] = "manual_include" if passes_filter else "manual_exclude"
    (OUTPUT_DIR / "chunks.json").write_text(json.dumps(all_chunks, indent=2))

    kept = _kept_chunks()
    existing = _find_chunk(kept, chunk_id)
    if passes_filter:
        if existing is None:
            # Freshly returned to analysis — add it back in a clean,
            # unclassified state rather than guessing at stale Stage 2/3
            # fields it never actually earned.
            fresh = dict(chunk)
            for f in ("chunk_type", "labels", "classification_confidence", "keep_classification",
                      "detected", "detection_confidence", "target_match", "keep_reason",
                      "extraction_done"):
                fresh[f] = [] if f == "labels" else None
            kept.append(fresh)
        else:
            existing["passes_filter"] = True
            existing["filter_reason"] = "manual_include"
        _save_kept_chunks(kept)
    else:
        if existing is not None:
            kept = [c for c in kept if c.get("chunk_id") != chunk_id]
            _save_kept_chunks(kept)
        _purge_records_for_chunk(chunk_id)

    return jsonify({"chunk_id": chunk_id, "passes_filter": passes_filter})


# --------------------------------------------------------------------------- #
# Settings: API key + model name for the three AI slots (cheap/main/validation)
# --------------------------------------------------------------------------- #
@app.route("/api/models", methods=["GET"])
def get_models():
    """The fixed list of valid model strings, used to populate the Settings
    page's model dropdowns — keeps model selection to known-good values
    instead of a free-text field a typo could silently break. Each model also
    carries its supported effort levels (empty list if the model doesn't
    support the effort parameter at all, e.g. Haiku), so the UI can grey out
    or hide the effort dropdown appropriately per model."""
    models = [
        {**m, "effort_levels": effort_levels_for(m["id"])}
        for m in AVAILABLE_MODELS
    ]
    return jsonify({"models": models})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(settings_store.load_masked())


@app.route("/api/settings", methods=["POST"])
def save_settings():
    data = request.get_json(force=True) or {}
    settings_store.save(data)
    return jsonify(settings_store.load_masked())


# --------------------------------------------------------------------------- #
# Prompts: the four (system, user) templates behind Stages 2-5, editable from
# the Settings page instead of hand-edited in prompts.py.
# --------------------------------------------------------------------------- #
@app.route("/api/prompts", methods=["GET"])
def get_prompts():
    return jsonify({"prompts": prompts_store.load(), "tokens": prompts_store.TOKENS})


@app.route("/api/prompts", methods=["POST"])
def save_prompts():
    data = request.get_json(force=True) or {}
    prompts_store.save(data)
    return jsonify({"prompts": prompts_store.load(), "tokens": prompts_store.TOKENS})


# --------------------------------------------------------------------------- #
# Shared: refuse to run an AI stage that isn't actually configured yet, rather
# than silently doing nothing (or letting an exception surface mid-loop).
# --------------------------------------------------------------------------- #
_STAGE_NAMES = {
    "classification": "Classification",
    "detection": "Detection",
    "paper_metadata": "Paper metadata",
    "extraction": "Extraction",
    "validation": "Validation",
}


def _ai_readiness_error(slot: str, stage: str):
    """Returns a human-readable error string if the given AI slot (cheap/main/
    validation) has no API key configured, or the given stage's prompt
    (classification/detection/extraction/validation) is still blank. Returns
    None if both are ready to actually call the model.
    """
    cfg = settings_store.load_model_config()
    stage_cfg = getattr(cfg, slot)
    if not stage_cfg.api_key:
        return (
            f"No API key set for the '{slot}' AI. Set it on the Settings page "
            f"(or via the {stage_cfg.api_key_env} environment variable) before "
            f"running {_STAGE_NAMES.get(stage, stage)}."
        )
    prompt = prompts_store.load().get(stage, {"system": "", "user": ""})
    if not prompt["system"].strip() and not prompt["user"].strip():
        return (
            f"The {_STAGE_NAMES.get(stage, stage)} prompt is blank. Write it on "
            f"the Settings page before running this stage."
        )
    return None


# --------------------------------------------------------------------------- #
# Cheap AI #1 — Stage 2 classification over already-kept chunks
# --------------------------------------------------------------------------- #
def _kept_chunks() -> list:
    return _load_json(OUTPUT_DIR / "chunks_kept.json", [])


def _save_kept_chunks(chunks: list) -> None:
    (OUTPUT_DIR / "chunks_kept.json").write_text(json.dumps(chunks, indent=2))


def _classifiable_chunks(chunks: list) -> list:
    """chunks_kept.json holds everything Stage 3a decided to keep, which
    includes two kinds of entries Stage 2 classification should never see:

    - the abstract chunk (is_abstract=True) — it's kept only so it can be
      handed to the main extraction AI as separate whole-paper context, not
      because it's a normal candidate chunk. It never needs a content-type
      label.
    - anything the heuristic filter actually dropped (passes_filter is
      False) — these shouldn't be in chunks_kept.json to begin with, but this
      guards against stale output from an older pipeline run still showing up
      (and getting classified) on this page.

    Filtered out here, defensively, rather than trusting every upstream
    writer of chunks_kept.json to have excluded them already.
    """
    return [
        c for c in chunks
        if not c.get("is_abstract") and c.get("passes_filter", True) is not False
    ]


# --------------------------------------------------------------------------- #
# Cross-worker progress state for the three long-running stage loops
# (classify/detect/extract), persisted to a small JSON file under the active
# project's output dir rather than a plain in-memory module-level dict.
#
# Gunicorn (see the deploy systemd unit's --workers flag) runs multiple
# worker PROCESSES, each with its own separate Python memory — a dict
# written to by the worker running a several-minute classify/detect/extract
# loop is invisible to whichever OTHER worker happens to answer a /progress
# poll or a /pause click. A sync worker can only handle one request at a
# time, so while the loop's worker is busy, literally every other request
# (including every progress poll) gets served by a different worker whose
# copy of that dict was never touched — the progress bar silently freezes at
# whatever that other worker's dict happened to hold, and a /pause click can
# silently no-op the same way. A small file under OUTPUT_DIR is on the same
# filesystem for every worker, so it's a shared source of truth regardless of
# which process answers which request. Writing it once per chunk is cheap —
# each chunk already costs a full LLM round-trip, dwarfing one small write.
# --------------------------------------------------------------------------- #
_CLASSIFY_PROGRESS_DEFAULT = {
    "running": False,
    "paused": False,      # set by /pause; the loop blocks between chunks while true
    "paper_id": None,
    "chunk_index": 0,     # 1-based position of the chunk currently being classified
    "total": 0,           # total chunks across all papers
    "classified": 0,      # chunks successfully classified so far
    "parse_failed": 0,    # chunks where the model answered but parsing failed
    "error": None,
}
_DETECT_PROGRESS_DEFAULT = {
    "running": False,
    "paper_id": None,
    "chunk_index": 0,
    "total": 0,
    "screened": 0,
    "error": None,
}
_EXTRACT_PROGRESS_DEFAULT = {
    "running": False,
    "paper_id": None,
    "chunk_index": 0,
    "total": 0,
    "extracted": 0,
    "records_found": 0,
    "error": None,
}


def _progress_path(stage: str) -> Path:
    return OUTPUT_DIR / f"_progress_{stage}.json"


def _read_progress(stage: str, default: dict) -> dict:
    return _load_json(_progress_path(stage), dict(default))


def _write_progress(stage: str, data: dict) -> None:
    _progress_path(stage).write_text(json.dumps(data))


@app.route("/api/cheap-ai/classify/progress", methods=["GET"])
def cheap_ai_classify_progress():
    return jsonify(_read_progress("classify", _CLASSIFY_PROGRESS_DEFAULT))


@app.route("/api/cheap-ai/classify/pause", methods=["POST"])
def cheap_ai_classify_pause():
    progress = _read_progress("classify", _CLASSIFY_PROGRESS_DEFAULT)
    if not progress["running"]:
        return jsonify({"error": "no classification run is currently in progress"}), 400
    progress["paused"] = True
    _write_progress("classify", progress)
    return jsonify(progress)


@app.route("/api/cheap-ai/classify/resume", methods=["POST"])
def cheap_ai_classify_resume():
    progress = _read_progress("classify", _CLASSIFY_PROGRESS_DEFAULT)
    progress["paused"] = False
    _write_progress("classify", progress)
    return jsonify(progress)


@app.route("/api/detection-ai/detect/progress", methods=["GET"])
def detection_ai_detect_progress():
    return jsonify(_read_progress("detect", _DETECT_PROGRESS_DEFAULT))


@app.route("/api/main-ai/extract/progress", methods=["GET"])
def main_ai_extract_progress():
    return jsonify(_read_progress("extract", _EXTRACT_PROGRESS_DEFAULT))


@app.route("/api/cheap-ai/papers", methods=["GET"])
def cheap_ai_papers():
    """Per-paper counts of kept chunks and how many have been classified so far.

    Excludes the abstract and anything heuristic-dropped — see
    _classifiable_chunks — so this page's totals/coverage % only ever reflect
    chunks Stage 2 actually needs to look at.
    """
    chunks = _classifiable_chunks(_kept_chunks())
    by_paper: dict = {}
    for c in chunks:
        stats = by_paper.setdefault(c["paper_id"], {"total": 0, "classified": 0})
        stats["total"] += 1
        if c.get("chunk_type"):
            stats["classified"] += 1
    out = [
        {"paper_id": pid, "total_chunks": s["total"], "classified_chunks": s["classified"]}
        for pid, s in sorted(by_paper.items())
    ]
    return jsonify({"papers": out})


@app.route("/api/cheap-ai/papers/<paper_id>/chunks", methods=["GET"])
def cheap_ai_paper_chunks(paper_id):
    chunks = [c for c in _classifiable_chunks(_kept_chunks()) if c["paper_id"] == paper_id]
    return jsonify({"paper_id": paper_id, "chunks": chunks})


@app.route("/api/cheap-ai/papers/<paper_id>/chunks/<path:chunk_id>/override", methods=["POST"])
def cheap_ai_chunk_override(paper_id, chunk_id):
    """Stage 2 manual edit: a human picks the correct content-type label from
    the same dropdown of ChunkType values the model itself must choose from.
    The manual label becomes authoritative (chunk_type AND labels, replacing
    whatever the model produced) — this is a correction, not another vote to
    average in. Downstream Stage 3/4 results for this one chunk are purged
    (a changed content type can change whether its earlier detection verdict
    still makes sense), scoped to just this chunk rather than a global
    re-run."""
    data = request.get_json(force=True) or {}
    chunk_type = data.get("chunk_type")
    valid_types = [t.value for t in ChunkType]
    if chunk_type not in valid_types:
        return jsonify({"error": f"chunk_type must be one of {valid_types}"}), 400

    kept = _kept_chunks()
    chunk = _find_chunk(kept, chunk_id)
    if chunk is None or chunk.get("paper_id") != paper_id:
        return jsonify({"error": "chunk not found"}), 404

    chunk["chunk_type"] = chunk_type
    chunk["labels"] = [chunk_type]
    chunk["classification_confidence"] = "manual"
    chunk["keep_classification"] = chunk_type not in ("other", "literature_review")
    chunk["classification_parse_failed"] = False
    # this chunk's own downstream verdicts no longer apply to the corrected label
    chunk["detected"] = None
    chunk["detection_confidence"] = None
    chunk["target_match"] = None
    chunk["keep_reason"] = None
    chunk["extraction_done"] = None
    _save_kept_chunks(kept)
    _purge_records_for_chunk(chunk_id)

    return jsonify({"chunk_id": chunk_id, "chunk_type": chunk_type})


@app.route("/api/cheap-ai/classify", methods=["POST"])
def cheap_ai_classify():
    """Run Stage 2 classification (the first cheap AI) over every kept chunk,
    using whichever cheap-model settings are saved on the Settings page.

    Refuses to run (400) if the cheap AI's API key isn't set or the
    classification prompt is still blank, rather than quietly doing nothing.
    """
    all_chunks = _kept_chunks()
    # Only chunks Stage 2 should actually judge — never the abstract (it's
    # reserved as separate whole-paper context for the main extraction AI)
    # and never anything the heuristic filter actually dropped. `classifiable`
    # entries are the same dict objects as in `all_chunks`, so mutating them
    # in place and saving `all_chunks` back keeps the abstract/dropped
    # entries untouched in chunks_kept.json.
    classifiable = _classifiable_chunks(all_chunks)
    if not classifiable:
        return jsonify({"error": "no classifiable kept chunks yet — run Analyze on the Corpus page first"}), 400

    err = _ai_readiness_error("cheap", "classification")
    if err:
        return jsonify({"error": err}), 400

    cfg = settings_store.load_model_config()
    llm = LLMClient(cfg)

    # Each chunk is a separate blocking API call, so a full run over hundreds
    # of chunks can take minutes. Save to disk every few chunks (not just at
    # the end) so /api/cheap-ai/papers reflects live progress, and keep the
    # shared "classify" progress file updated chunk-by-chunk (see
    # _write_progress above) so the page can show a live progress bar with
    # the paper currently being worked on, rather than looking stuck for the
    # whole request.
    SAVE_EVERY = 5
    classified = 0
    parse_failures = 0
    total = len(classifiable)
    progress = dict(_CLASSIFY_PROGRESS_DEFAULT)
    progress.update(
        running=True, paused=False, paper_id=classifiable[0]["paper_id"] if classifiable else None,
        chunk_index=0, total=total, classified=0, parse_failed=0, error=None,
    )
    _write_progress("classify", progress)
    try:
        for i, c in enumerate(classifiable):
            # Block between chunks while paused, rather than mid-API-call —
            # a paused run always stops at a clean chunk boundary. Re-read
            # the shared file (not the local `progress` var) for the pause
            # check, since the /pause click that set it may have been
            # answered by a different gunicorn worker than this one. Save
            # progress to disk as soon as the pause takes effect so a
            # paused-then-abandoned run doesn't lose anything.
            if _read_progress("classify", _CLASSIFY_PROGRESS_DEFAULT).get("paused"):
                _save_kept_chunks(all_chunks)
                progress["paused"] = True
                _write_progress("classify", progress)
                while _read_progress("classify", _CLASSIFY_PROGRESS_DEFAULT).get("paused"):
                    time.sleep(0.5)
                progress["paused"] = False
            progress["paper_id"] = c["paper_id"]
            progress["chunk_index"] = i + 1
            _write_progress("classify", progress)
            try:
                result = llm.classify_chunk(c["text"])
            except Exception as e:
                _save_kept_chunks(all_chunks)  # keep whatever progress was made
                progress["error"] = str(e)
                progress["running"] = False
                _write_progress("classify", progress)
                return jsonify({"error": str(e), "classified": classified, "total": total}), 502
            if result:
                labels = []
                for lbl in result.get("labels") or []:
                    try:
                        labels.append(ChunkType(lbl).value)
                    except ValueError:
                        continue
                primary = result.get("primary")
                try:
                    primary_value = ChunkType(primary).value if primary else (labels[0] if labels else ChunkType.OTHER.value)
                except ValueError:
                    primary_value = labels[0] if labels else ChunkType.OTHER.value
                c["chunk_type"] = primary_value
                c["labels"] = labels
                c["classification_confidence"] = result.get("confidence")
                c["keep_classification"] = result.get("keep")
                c["classification_parse_failed"] = False
                classified += 1
                progress["classified"] = classified
            else:
                # llm.classify_chunk() returns None in exactly two cases: the
                # prompt is blank, or the model responded but the response
                # didn't parse into the expected shape. _ai_readiness_error
                # above already refused to run this whole request if the
                # prompt were blank, so by the time we're here None can only
                # mean the second case — the model answered but Stage 2's
                # parser couldn't use it (malformed/truncated JSON, refusal
                # text, etc.). That used to be silently indistinguishable
                # from "never attempted" (chunk_type stays null either way);
                # flagged explicitly here so the chunk shows up in the UI as
                # a parse failure to investigate/retry rather than looking
                # identical to a chunk that just hasn't been reached yet.
                c["classification_parse_failed"] = True
                parse_failures += 1
                progress["parse_failed"] = parse_failures
            if (i + 1) % SAVE_EVERY == 0:
                _save_kept_chunks(all_chunks)
                _write_progress("classify", progress)

        _save_kept_chunks(all_chunks)
        # Classification just (re-)ran, so any prior detection verdicts are
        # stale — clear them so detection_ai_done correctly recomputes to
        # False (re-greying Detection AI) until it's re-run against the new
        # classification.
        _discard_detection_results()
        return jsonify({"total": total, "classified": classified, "parse_failed": parse_failures})
    finally:
        progress["running"] = False
        _write_progress("classify", progress)


# --------------------------------------------------------------------------- #
# Cheap AI #2 — Stage 3 detection: does this (already heuristic-kept) chunk
# actually contain a point estimate for one of the user's targets?
# --------------------------------------------------------------------------- #
def _detection_visible_chunks(chunks: list) -> list:
    """Chunks the Detection AI page should display: the same set Cheap AI
    classified (not abstract, not heuristic-dropped) — but, unlike
    _classifiable_chunks, INCLUDING chunks detection itself has since
    screened out (passes_filter False with filter_reason
    'cheap_llm_negative' or 'manual_exclude'). Otherwise a chunk would vanish
    from this page the instant it's screened negative, and a human could
    never see it again to correct a wrong "no" via the keep_reason dropdown.
    Chunks the HEURISTIC filter dropped (any other filter_reason) never
    reached detection at all and are still correctly excluded."""
    return [
        c for c in chunks
        if not c.get("is_abstract")
        and (c.get("passes_filter", True) is not False
             or c.get("filter_reason") in ("cheap_llm_negative", "manual_exclude"))
    ]


@app.route("/api/detection-ai/papers", methods=["GET"])
def detection_ai_papers():
    """Per-paper counts of kept chunks and how many have been screened so far.

    Excludes the abstract and anything heuristic-dropped — see
    _detection_visible_chunks — same reasoning as the Cheap AI
    (classification) page: the abstract is reserved as separate whole-paper
    context for the main extraction AI and should never be screened/dropped
    here. Unlike Cheap AI, chunks detection itself screened negative stay
    visible (see _detection_visible_chunks) so they can be corrected.
    """
    chunks = _detection_visible_chunks(_kept_chunks())
    by_paper: dict = {}
    for c in chunks:
        stats = by_paper.setdefault(c["paper_id"], {"total": 0, "screened": 0})
        stats["total"] += 1
        if c.get("detected") is not None:
            stats["screened"] += 1
    out = [
        {"paper_id": pid, "total_chunks": s["total"], "screened_chunks": s["screened"]}
        for pid, s in sorted(by_paper.items())
    ]
    return jsonify({"papers": out})


@app.route("/api/detection-ai/papers/<paper_id>/chunks", methods=["GET"])
def detection_ai_paper_chunks(paper_id):
    chunks = [c for c in _detection_visible_chunks(_kept_chunks()) if c["paper_id"] == paper_id]
    return jsonify({"paper_id": paper_id, "chunks": chunks})


@app.route("/api/detection-ai/papers/<paper_id>/chunks/<path:chunk_id>/override", methods=["POST"])
def detection_ai_chunk_override(paper_id, chunk_id):
    """Stage 3 manual edit: a human picks the correct keep_reason from the
    same dropdown of values the model itself must choose from. keep_reason
    doubles as the detected yes/no call — "off_target" and
    "no_estimate_signal" both mean "no", every other reason means "yes" —
    same rule the model itself follows, so picking a reason IS picking
    detected. A manual "no" drops the chunk from the kept set the same way a
    cheap-LLM "no" does (passes_filter=False); a manual "yes" undoes that.
    Extraction records already built from this chunk are purged, scoped to
    just this chunk."""
    data = request.get_json(force=True) or {}
    keep_reason = data.get("keep_reason")
    if keep_reason not in LLMClient._KEEP_REASONS:
        return jsonify({"error": f"keep_reason must be one of {list(LLMClient._KEEP_REASONS)}"}), 400

    kept = _kept_chunks()
    chunk = _find_chunk(kept, chunk_id)
    if chunk is None or chunk.get("paper_id") != paper_id:
        return jsonify({"error": "chunk not found"}), 404

    detected = keep_reason not in ("off_target", "no_estimate_signal")
    chunk["keep_reason"] = keep_reason
    chunk["detected"] = detected
    chunk["detection_confidence"] = "manual"
    chunk["passes_filter"] = detected
    chunk["filter_reason"] = None if detected else "manual_exclude"
    chunk["extraction_done"] = None
    _save_kept_chunks(kept)
    _purge_records_for_chunk(chunk_id)

    return jsonify({"chunk_id": chunk_id, "keep_reason": keep_reason, "detected": detected})


@app.route("/api/detection-ai/detect", methods=["POST"])
def detection_ai_detect():
    """Run Stage 3 detection (the second cheap AI) over every kept chunk: a
    binary "does this plausibly contain a point estimate for one of the user's
    targets" screen, using whichever cheap-model settings are saved on the
    Settings page. Chunks the model says no to are marked passes_filter=False /
    filter_reason="cheap_llm_negative" — the same outcome Stage 3b would
    produce in the full pipeline run — so the Cheap AI / Filter results pages
    and any later extraction step all see a consistent kept set.

    Refuses to run (400) if the cheap AI's API key isn't set or the detection
    prompt is still blank, rather than quietly doing nothing.
    """
    all_chunks = _kept_chunks()
    # Same reasoning as Cheap AI classification: never screen the abstract
    # (it's exempt/reserved context, not a normal candidate chunk) or
    # anything already dropped. `classifiable` shares dict objects with
    # `all_chunks`, so in-place edits here are still captured when
    # `all_chunks` is saved back.
    classifiable = _classifiable_chunks(all_chunks)
    if not classifiable:
        return jsonify({"error": "no classifiable kept chunks yet — run Analyze on the Corpus page first"}), 400

    err = _ai_readiness_error("cheap", "detection")
    if err:
        return jsonify({"error": err}), 400

    targets = _load_json(TARGETS_PATH, {})
    cfg = settings_store.load_model_config()
    llm = LLMClient(cfg)

    # Same reasoning as Cheap AI classification: this can run for minutes
    # over hundreds of chunks, so save to disk every few chunks (not just at
    # the end) and keep the shared "detect" progress file updated
    # chunk-by-chunk so the page can show a live progress bar with the paper
    # currently being worked on.
    SAVE_EVERY = 5
    screened = 0
    dropped = 0
    total = len(classifiable)
    progress = dict(_DETECT_PROGRESS_DEFAULT)
    progress.update(
        running=True, paper_id=classifiable[0]["paper_id"] if classifiable else None,
        chunk_index=0, total=total, screened=0, error=None,
    )
    _write_progress("detect", progress)
    try:
        for i, c in enumerate(classifiable):
            progress["paper_id"] = c["paper_id"]
            progress["chunk_index"] = i + 1
            _write_progress("detect", progress)
            # Hand Stage 2's own output for this chunk to Stage 3 as context —
            # a high-confidence "regression_table"/"result_text" classification
            # is a strong prior that a point estimate is present, while "other"
            # or a keep:false "literature_review" call is a strong prior it isn't.
            classification_context = {
                "labels": c.get("labels") or ([c["chunk_type"]] if c.get("chunk_type") else []),
                "primary": c.get("chunk_type"),
                "confidence": c.get("classification_confidence"),
                "keep": c.get("keep_classification"),
            }
            try:
                result = llm.detect_estimate(c["text"], targets, classification_context)
            except Exception as e:
                _save_kept_chunks(all_chunks)  # keep whatever progress was made
                progress["error"] = str(e)
                progress["running"] = False
                _write_progress("detect", progress)
                return jsonify({"error": str(e), "screened": screened, "total": total}), 502
            if result is not None:
                detected = result.get("detected")
                c["detected"] = detected
                c["detection_confidence"] = result.get("confidence")
                c["target_match"] = result.get("target_match")
                c["keep_reason"] = result.get("keep_reason")
                screened += 1
                progress["screened"] = screened
                if detected is False:
                    c["passes_filter"] = False
                    c["filter_reason"] = "cheap_llm_negative"
                    dropped += 1
            if (i + 1) % SAVE_EVERY == 0:
                _save_kept_chunks(all_chunks)
                _write_progress("detect", progress)

        _save_kept_chunks(all_chunks)
        # Detection just (re-)ran, so any prior extraction records are stale —
        # clear them so main_ai_done correctly recomputes to False (re-greying
        # Main AI) until it's re-run against the new detected set.
        _discard_extraction_results()
        # No explicit "mark done" needed — _load_state() recomputes
        # detection_ai_done directly from the 'detected' fields just written.
        return jsonify({"total": total, "screened": screened, "dropped": dropped})
    finally:
        progress["running"] = False
        _write_progress("detect", progress)


# --------------------------------------------------------------------------- #
# Main AI — Stage 4 extraction: pull structured point-estimate records out of
# every chunk detection actually said yes to. Runs a Stage 4a paper-metadata
# pass first (once per paper, not per chunk — see meta_pipeline.pipeline's
# own version of this), then the per-chunk extraction call, writing results
# to records.json rather than back onto chunks_kept.json (a chunk can now
# yield zero, one, or several records, so it doesn't fit as a single field).
# --------------------------------------------------------------------------- #
@app.route("/api/main-ai/papers", methods=["GET"])
def main_ai_papers():
    """Per-paper counts of extractable chunks and how many have been run
    through extraction so far (see _extractable_chunks)."""
    extractable = _extractable_chunks(_kept_chunks())
    by_paper: dict = {}
    for c in extractable:
        stats = by_paper.setdefault(c["paper_id"], {"total": 0, "extracted": 0})
        stats["total"] += 1
        if c.get("extraction_done"):
            stats["extracted"] += 1
    out = [
        {"paper_id": pid, "total_chunks": s["total"], "extracted_chunks": s["extracted"]}
        for pid, s in sorted(by_paper.items())
    ]
    return jsonify({"papers": out})


@app.route("/api/main-ai/records", methods=["GET"])
def main_ai_records():
    return jsonify({"records": _load_json(OUTPUT_DIR / "records.json", [])})


# Which enum fields validate against which vocab — mirrors the same
# controlled vocabularies LLMClient itself enforces on the model's output, so
# a manual edit can't introduce a value the model would never have been
# allowed to produce.
_RECORD_ENUM_FIELDS = {
    "match_type": LLMClient._RECORD_MATCH_TYPES,
    "variable_role": LLMClient._VARIABLE_ROLES,
    "estimate_type": LLMClient._ESTIMATE_TYPES,
    "specification_status": LLMClient._SPEC_STATUSES,
    "unit_source": LLMClient._UNIT_SOURCES,
    "source_type": LLMClient._SOURCE_TYPES,
    "elasticity_transformation_type": LLMClient._TRANSFORMATION_TYPES,
    "target_food_group": LLMClient._FOOD_GROUPS,
    "target_cross_price_food_group": LLMClient._FOOD_GROUPS,
}
_RECORD_NUMERIC_FIELDS = {"coefficient", "standard_error", "p_value", "n_obs", "n_units"}
_RECORD_INT_FIELDS = {"n_obs", "n_units"}
_RECORD_BOOL_FIELDS = {
    "elasticity_is_raw", "requires_review",
    "standard_error_reported", "confidence_interval_reported", "p_value_reported",
}
_RECORD_TEXT_FIELDS = {
    "paper_id", "target_elasticity_type", "target_product", "target_cross_price_product",
    "paper_elasticity_wording_raw", "paper_product_wording_raw", "paper_cross_price_product_wording_raw",
    "target_match_justification", "significance_stars", "baseline_evidence",
    "model_type", "countries_region", "frequency", "data_source", "elasticity_unit",
}


def _blank_record(paper_id: str) -> dict:
    """A fully-shaped, empty extraction record — every field a normal
    LLM-produced record would have, just all null/default — so a manually
    added row goes through exactly the same edit form and rendering path as
    a machine-extracted one instead of needing special-cased handling."""
    return {
        "paper_id": paper_id or "",
        "estimate_id": f"manual::{uuid.uuid4().hex[:10]}",
        "source_chunk_id": None,
        "target_elasticity_type": None, "target_product": None, "target_cross_price_product": None,
        "target_food_group": None, "target_cross_price_food_group": None,
        "paper_elasticity_wording_raw": None, "paper_product_wording_raw": None,
        "paper_cross_price_product_wording_raw": None,
        "match_type": None, "target_match_justification": None,
        "variable_role": None, "estimate_type": None,
        "coefficient": None, "standard_error": None, "standard_error_reported": None,
        "p_value": None, "p_value_reported": None,
        "confidence_interval": None, "confidence_interval_reported": None,
        "test_statistic": None, "significance_stars": None,
        "elasticity_is_raw": None, "elasticity_transformation_type": None, "elasticity_unit": None,
        "unit_source": None, "specification_status": None, "baseline_evidence": None,
        "model_type": None, "countries_region": None, "frequency": None,
        "time_period": {"start": None, "end": None},
        "n_obs": None, "n_units": None, "data_source": None, "source_type": None,
        "source_location": {"page": None, "table": None, "text_anchor": None, "row": None, "column": None},
        "table_complete": None, "pages_used": [],
        "requires_review": False, "review_reason": [],
        "manually_edited": True, "manually_added": True,
    }


@app.route("/api/main-ai/records", methods=["PUT"])
def main_ai_add_record():
    """Add a blank record by hand (not derived from any chunk) — for cases
    the pipeline missed entirely rather than got wrong. Returns the new
    record immediately so the frontend can drop straight into edit mode on
    it, same as clicking Edit on any other row."""
    data = request.get_json(force=True, silent=True) or {}
    records = _load_json(OUTPUT_DIR / "records.json", [])
    rec = _blank_record(data.get("paper_id"))
    records.append(rec)
    (OUTPUT_DIR / "records.json").write_text(json.dumps(records, indent=2))
    return jsonify({"record": rec})


@app.route("/api/main-ai/records/<path:estimate_id>", methods=["DELETE"])
def main_ai_delete_record(estimate_id):
    """Drop a record entirely — for extraction noise (a row that shouldn't
    exist at all) as opposed to a wrong value in an otherwise-real row,
    which is what the edit endpoint below is for."""
    records = _load_json(OUTPUT_DIR / "records.json", [])
    kept = [r for r in records if r.get("estimate_id") != estimate_id]
    if len(kept) == len(records):
        return jsonify({"error": "record not found"}), 404
    (OUTPUT_DIR / "records.json").write_text(json.dumps(kept, indent=2))
    return jsonify({"ok": True})


@app.route("/api/main-ai/records/<path:estimate_id>", methods=["POST"])
def main_ai_update_record(estimate_id):
    """Stage 4 manual edit: directly edit any field of an already-extracted
    record. Accepts a partial update (only the fields the user actually
    changed) — unknown keys are silently ignored rather than erroring, so the
    frontend can always submit its whole edit form without diffing it first.
    Enum fields are validated against the same vocab the model itself must
    use; numeric/boolean fields are coerced defensively. Edited records are
    flagged manually_edited so they're visually distinguishable from
    machine-extracted ones."""
    data = request.get_json(force=True) or {}
    records = _load_json(OUTPUT_DIR / "records.json", [])
    rec = next((r for r in records if r.get("estimate_id") == estimate_id), None)
    if rec is None:
        return jsonify({"error": "record not found"}), 404

    for key, value in data.items():
        if key in _RECORD_ENUM_FIELDS:
            if value is not None and value not in _RECORD_ENUM_FIELDS[key]:
                return jsonify({"error": f"{key} must be one of {list(_RECORD_ENUM_FIELDS[key])} or null"}), 400
            rec[key] = value
        elif key in _RECORD_NUMERIC_FIELDS:
            if value is None or value == "":
                rec[key] = None
            else:
                try:
                    rec[key] = int(value) if key in _RECORD_INT_FIELDS else float(value)
                except (TypeError, ValueError):
                    return jsonify({"error": f"{key} must be a number"}), 400
        elif key in _RECORD_BOOL_FIELDS:
            rec[key] = bool(value) if value is not None else None
        elif key in _RECORD_TEXT_FIELDS:
            rec[key] = value if (value is None or isinstance(value, str)) else str(value)
        elif key == "confidence_interval":
            if value is None:
                rec[key] = None
            elif isinstance(value, list) and len(value) == 2:
                try:
                    rec[key] = [float(value[0]), float(value[1])]
                except (TypeError, ValueError):
                    return jsonify({"error": "confidence_interval must be [low, high]"}), 400
            else:
                return jsonify({"error": "confidence_interval must be [low, high] or null"}), 400
        elif key == "time_period":
            if isinstance(value, dict):
                try:
                    rec[key] = {
                        "start": int(value["start"]) if value.get("start") not in (None, "") else None,
                        "end": int(value["end"]) if value.get("end") not in (None, "") else None,
                    }
                except (TypeError, ValueError):
                    return jsonify({"error": "time_period start/end must be years or null"}), 400
        elif key == "test_statistic":
            if value is None:
                rec[key] = None
            elif isinstance(value, dict) and value.get("type") and value.get("value") not in (None, ""):
                try:
                    rec[key] = {"type": value["type"], "value": float(value["value"])}
                except (TypeError, ValueError):
                    return jsonify({"error": "test_statistic.value must be a number"}), 400
            else:
                rec[key] = None
        elif key == "review_reason":
            if isinstance(value, list):
                rec[key] = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, str):
                rec[key] = [v.strip() for v in value.split(",") if v.strip()]
        elif key in ("row", "column"):
            loc = rec.setdefault("source_location", {}) or {}
            rec["source_location"] = loc
            loc[key] = value or None

    rec["manually_edited"] = True
    (OUTPUT_DIR / "records.json").write_text(json.dumps(records, indent=2))
    return jsonify({"record": rec})


def _abstracts_by_paper() -> dict:
    abstracts = _load_json(OUTPUT_DIR / "abstracts.json", [])
    return {a["paper_id"]: a.get("abstract") for a in abstracts}


def _methodology_text_by_paper(chunks: list) -> dict:
    """Concatenated methodology/data_description chunk text per paper, for
    the Stage 4a paper-metadata pass. Pulled from chunks_kept.json — unlike a
    single Pipeline.run() call (which has the full unfiltered+classified
    chunk set in memory at once), the webapp's incremental per-page flow only
    ever classifies/persists chunks that made it into the kept set, so a
    methodology sentence with no estimate signal that was dropped before
    Cheap AI ever saw it simply isn't available here. Same reasoning as
    meta_pipeline.pipeline.Pipeline.extract_paper_metadata, applied to
    whatever's actually on hand in this flow."""
    by_paper: dict = {}
    for c in chunks:
        labels = c.get("labels") or ([c["chunk_type"]] if c.get("chunk_type") else [])
        if "methodology" in labels or "data_description" in labels:
            by_paper.setdefault(c["paper_id"], []).append(c["text"])
    return {pid: "\n\n".join(texts)[:12000] for pid, texts in by_paper.items()}


def _record_from_estimate(raw: dict, chunk: dict, idx: int, meta: dict) -> dict:
    """Dict-based equivalent of meta_pipeline.pipeline.Pipeline._record_from_raw
    + _backfill_paper_metadata, for the webapp's incremental per-chunk
    extraction flow (working against chunk JSON dicts, not Chunk dataclass
    instances). Provenance (page/table/text_anchor/table_complete/pages_used)
    always comes from the chunk, never the model; row/column are the model's
    to set. Paper-level fields (model_type, countries_region, time_period,
    frequency, n_obs, n_units, data_source) are backfilled from `meta` only
    where the model's own per-chunk output left them null."""
    rec = dict(raw)
    rec["paper_id"] = chunk["paper_id"]
    rec["estimate_id"] = f"{chunk['chunk_id']}::est{idx}"
    rec["source_chunk_id"] = chunk["chunk_id"]

    src = chunk.get("source_location") or {}
    rec["source_location"] = {
        "page": src.get("page"),
        "table": src.get("table"),
        "text_anchor": src.get("text_anchor"),
        "row": raw.get("row") if isinstance(raw.get("row"), str) else src.get("row"),
        "column": raw.get("column") if isinstance(raw.get("column"), str) else src.get("column"),
    }
    rec.pop("row", None)
    rec.pop("column", None)
    rec["table_complete"] = chunk.get("table_complete")
    rec["pages_used"] = chunk.get("pages_used") or []

    meta = meta or {}
    for field in ("model_type", "countries_region", "frequency", "n_obs", "n_units", "data_source"):
        if rec.get(field) is None:
            rec[field] = meta.get(field)
    tp = rec.get("time_period")
    if not tp or (tp.get("start") is None and tp.get("end") is None):
        rec["time_period"] = meta.get("time_period") or {"start": None, "end": None}
    return rec


def _decimal_places(v: float) -> int:
    s = f"{v:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def _is_rounded_duplicate(rounded: Optional[float], raw: Optional[float]) -> bool:
    """True if `rounded` looks like `raw` restated at lower precision — e.g.
    rounded=-0.35, raw=-0.353. Requires `rounded` to actually carry fewer
    decimal places than `raw` (otherwise two independently-reported values
    that coincidentally match to N places would be flagged as duplicates)."""
    if not isinstance(rounded, (int, float)) or not isinstance(raw, (int, float)):
        return False
    d_rounded, d_raw = _decimal_places(rounded), _decimal_places(raw)
    if d_rounded >= d_raw:
        return False
    return abs(round(raw, d_rounded) - rounded) < 1e-9


def _dedupe_rounded_records(records: list) -> "tuple[list, int]":
    """Text and a table in the same paper sometimes restate the identical
    elasticity twice at different precision — e.g. prose says "-0.35" while
    the underlying table says "-0.353". Since chunking sends the prose and
    the table to extraction as separate excerpts, each can independently
    yield its own record for what is really a single estimate (the
    extraction prompt already prevents this *within* one excerpt, but can't
    see across chunks). When two records agree on everything that identifies
    a distinct estimate (paper, elasticity type, product, cross-price
    product, variable role, estimate type) and one's coefficient is just a
    rounding of the other's, keep only the more precise (raw) one.

    Records a human has touched (manually_edited/manually_added) are never
    auto-dropped — only genuinely machine-extracted duplicates are cleaned
    up automatically. Returns (deduped_records, number_removed)."""
    def identity_key(r):
        return (
            r.get("paper_id"), r.get("target_elasticity_type"), r.get("target_product"),
            r.get("target_cross_price_product"), r.get("variable_role"), r.get("estimate_type"),
        )

    groups: Dict[tuple, list] = {}
    for r in records:
        groups.setdefault(identity_key(r), []).append(r)

    dropped_ids = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        survivors = list(group)
        i = 0
        while i < len(survivors):
            a = survivors[i]
            if a.get("manually_edited") or a.get("manually_added"):
                i += 1
                continue
            dropped_this_round = False
            for j, b in enumerate(survivors):
                if i == j or b.get("manually_edited") or b.get("manually_added"):
                    continue
                if _is_rounded_duplicate(a.get("coefficient"), b.get("coefficient")):
                    dropped_ids.add(a["estimate_id"])
                    survivors.pop(i)
                    dropped_this_round = True
                    break
            if not dropped_this_round:
                i += 1

    if not dropped_ids:
        return records, 0
    return [r for r in records if r.get("estimate_id") not in dropped_ids], len(dropped_ids)


@app.route("/api/main-ai/dedupe", methods=["POST"])
def main_ai_dedupe():
    """Standalone cleanup for records already sitting in records.json — runs
    the same rounded-duplicate check the extraction endpoint applies
    automatically, without calling the LLM again. Useful right after this
    fix ships, for data extracted before it existed."""
    records = _load_json(OUTPUT_DIR / "records.json", [])
    deduped, removed = _dedupe_rounded_records(records)
    if removed:
        (OUTPUT_DIR / "records.json").write_text(json.dumps(deduped, indent=2))
    return jsonify({"removed": removed, "records": len(deduped)})


@app.route("/api/main-ai/extract", methods=["POST"])
def main_ai_extract():
    """Run Stage 4 extraction (the main/expensive AI) over every chunk Stage 3
    detection said yes to. Refuses to run (400) if the main AI's API key isn't
    set or the extraction prompt is blank. The paper-metadata pass (Stage 4a)
    is skipped silently (not an error) if its own prompt is blank — extraction
    still works fine without it, just leaves paper-level fields null unless a
    specific chunk states them.
    """
    all_chunks = _kept_chunks()
    extractable = _extractable_chunks(all_chunks)
    if not extractable:
        return jsonify({"error": "no chunks have passed detection yet — run Detection AI first"}), 400

    err = _ai_readiness_error("main", "extraction")
    if err:
        return jsonify({"error": err}), 400

    targets = _load_json(TARGETS_PATH, {})
    ontology = _load_json(ONTOLOGY_PATH, {})
    cfg = settings_store.load_model_config()
    llm = LLMClient(cfg)

    abstracts = _abstracts_by_paper()
    methodology_text = _methodology_text_by_paper(all_chunks)
    paper_metadata_prompt = prompts_store.load().get("paper_metadata", {})
    paper_metadata_configured = bool(
        paper_metadata_prompt.get("system", "").strip() or paper_metadata_prompt.get("user", "").strip()
    )
    paper_metadata: dict = {}
    if paper_metadata_configured:
        for paper_id in sorted({c["paper_id"] for c in extractable}):
            try:
                meta = llm.extract_paper_metadata(abstracts.get(paper_id), methodology_text.get(paper_id, ""))
            except Exception:
                meta = None
            if meta:
                paper_metadata[paper_id] = meta

    # Extraction calls the main (usually pricier/slower) model, so save more
    # eagerly than the cheap stages do — same reasoning, smaller batch size.
    SAVE_EVERY = 3
    extracted_chunks = 0
    total = len(extractable)
    records = _load_json(OUTPUT_DIR / "records.json", [])
    progress = dict(_EXTRACT_PROGRESS_DEFAULT)
    progress.update(
        running=True, paper_id=extractable[0]["paper_id"] if extractable else None,
        chunk_index=0, total=total, extracted=0, records_found=len(records), error=None,
    )
    _write_progress("extract", progress)
    try:
        for i, c in enumerate(extractable):
            progress["paper_id"] = c["paper_id"]
            progress["chunk_index"] = i + 1
            _write_progress("extract", progress)
            screening = {
                "labels": c.get("labels") or ([c["chunk_type"]] if c.get("chunk_type") else []),
                "primary": c.get("chunk_type"),
                "classification_confidence": c.get("classification_confidence"),
                "keep_classification": c.get("keep_classification"),
                "detected": c.get("detected"),
                "detection_confidence": c.get("detection_confidence"),
                "target_match": c.get("target_match"),
                "keep_reason": c.get("keep_reason"),
            }
            try:
                raw_estimates = llm.extract(
                    c["text"], targets, ontology,
                    abstract=abstracts.get(c["paper_id"]),
                    screening=screening,
                )
            except Exception as e:
                _save_kept_chunks(all_chunks)
                (OUTPUT_DIR / "records.json").write_text(json.dumps(records, indent=2))
                progress["error"] = str(e)
                progress["running"] = False
                _write_progress("extract", progress)
                return jsonify({"error": str(e), "extracted": extracted_chunks, "total": total}), 502

            # Drop any earlier records sourced from this exact chunk before
            # adding fresh ones, so re-running extraction over an
            # already-processed chunk never duplicates rows.
            records = [r for r in records if r.get("source_chunk_id") != c["chunk_id"]]
            for idx, raw in enumerate(raw_estimates or []):
                records.append(_record_from_estimate(raw, c, idx, paper_metadata.get(c["paper_id"])))

            c["extraction_done"] = True
            extracted_chunks += 1
            progress["extracted"] = extracted_chunks
            progress["records_found"] = len(records)
            if (i + 1) % SAVE_EVERY == 0:
                _save_kept_chunks(all_chunks)
                (OUTPUT_DIR / "records.json").write_text(json.dumps(records, indent=2))
                _write_progress("extract", progress)

        # Text and a table chunk from the same paper can each independently
        # yield a record for the same real estimate at different precision
        # (see _dedupe_rounded_records) — clean that up across the whole
        # merged set now that every chunk has been processed, not per-chunk.
        records, deduped_count = _dedupe_rounded_records(records)

        _save_kept_chunks(all_chunks)
        (OUTPUT_DIR / "records.json").write_text(json.dumps(records, indent=2))
        # No explicit "mark done" needed — _load_state() recomputes
        # main_ai_done directly from the 'extraction_done' fields just written.
        return jsonify({
            "total": total, "extracted": extracted_chunks, "records": len(records),
            "duplicates_removed": deduped_count,
        })
    finally:
        progress["running"] = False
        _write_progress("extract", progress)


# --------------------------------------------------------------------------- #
# Stage 5 — Regression: meta-analysis over the extracted records table. Each
# run (a DV/IV choice + resulting fitted model) is appended to
# regression_runs.json under the active project's output dir, so several
# specifications can be compared side by side like columns (1), (2), (3) of
# a real paper's regression table, rather than only ever showing one run.
# --------------------------------------------------------------------------- #
def _load_regression_runs() -> list:
    return _load_json(OUTPUT_DIR / "regression_runs.json", [])


def _save_regression_runs(runs: list) -> None:
    (OUTPUT_DIR / "regression_runs.json").write_text(json.dumps(runs, indent=2))


@app.route("/api/regression/fields", methods=["GET"])
def regression_fields():
    return jsonify(regression.field_catalog())


@app.route("/api/regression/runs", methods=["GET"])
def regression_runs():
    return jsonify({"runs": _load_regression_runs()})


@app.route("/api/regression/run", methods=["POST"])
def regression_run():
    data = request.get_json(force=True, silent=True) or {}
    dv = data.get("dv")
    ivs = data.get("ivs") or []
    weight_by_precision = bool(data.get("weight_by_precision"))
    filters = data.get("filters") or {}
    label = (data.get("label") or "").strip()

    if not dv:
        return jsonify({"error": "Choose a dependent variable."}), 400
    if not ivs:
        return jsonify({"error": "Choose at least one independent variable."}), 400

    try:
        import pandas  # noqa: F401
        import statsmodels  # noqa: F401
    except ImportError:
        return jsonify({
            "error": "This feature needs the 'pandas' and 'statsmodels' packages. "
                     "Install them with: pip install pandas statsmodels"
        }), 400

    records = _load_json(OUTPUT_DIR / "records.json", [])
    try:
        result = regression.run_regression(
            records, dv, ivs, weight_by_precision=weight_by_precision, filters=filters,
        )
    except regression.RegressionError as e:
        return jsonify({"error": str(e)}), 400

    runs = _load_regression_runs()
    run_id = f"run_{int(time.time() * 1000)}_{len(runs) + 1}"
    entry = {
        "id": run_id,
        "label": label or f"({len(runs) + 1})",
        "created_at": time.time(),
        **result,
    }
    runs.append(entry)
    _save_regression_runs(runs)
    return jsonify(entry)


@app.route("/api/regression/runs/<run_id>", methods=["DELETE"])
def regression_delete_run(run_id):
    runs = _load_regression_runs()
    kept = [r for r in runs if r.get("id") != run_id]
    if len(kept) == len(runs):
        return jsonify({"error": "Run not found."}), 404
    _save_regression_runs(kept)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# Dashboard — the final "key findings" page: corpus/extraction summary
# statistics plus whatever regression runs are already saved (it reads
# regression_runs.json but doesn't run anything itself — Regression is where
# a model actually gets fit; Dashboard is presentation-only).
# --------------------------------------------------------------------------- #
def _top_counts(records: list, field: str, limit: int = 12) -> list:
    """[{value, count}, ...] for the most common non-empty values of `field`
    across records, most common first — used for the coverage breakdowns
    (products, elasticity types, countries, etc.) on the dashboard."""
    counter = Counter(r.get(field) for r in records if r.get(field))
    return [{"value": v, "count": c} for v, c in counter.most_common(limit)]


def _numeric_summary(records: list, field: str) -> dict:
    vals = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def _multi_query_param(name: str) -> list:
    """Reads a filter param that may arrive either as repeated query keys
    (?elasticity_type=a&elasticity_type=b) or as one comma-separated value
    (?elasticity_type=a,b) — the dashboard's checkbox filter sends the
    former, but this keeps the endpoint forgiving either way. Empty/blank
    entries are dropped; an empty result means "no filter" (show all)."""
    out = []
    for raw in request.args.getlist(name):
        out.extend(v.strip() for v in raw.split(",") if v.strip())
    return out


@app.route("/api/dashboard/summary", methods=["GET"])
def dashboard_summary():
    """Summary stats over the extracted records, optionally restricted to
    one or more of the elasticity types / products the user originally
    asked for on the upload page (targets.json) — e.g. only "Income
    elasticity" records, or only "Maize" — via repeated ?elasticity_type=
    and ?product= query params. Chunk/paper funnel counts (papers_total,
    chunks_*) describe the corpus as a whole and aren't filtered, since
    they're not meaningfully sliceable by a target that's only assigned
    once extraction runs; everything computed from records.json is."""
    kept = _kept_chunks()
    classifiable = _classifiable_chunks(kept)
    extractable = _extractable_chunks(classifiable)
    all_records = _load_json(OUTPUT_DIR / "records.json", [])

    elasticity_filter = set(_multi_query_param("elasticity_type"))
    product_filter = set(_multi_query_param("product"))
    food_group_filter = set(_multi_query_param("food_group"))
    records = [
        r for r in all_records
        if (not elasticity_filter or r.get("target_elasticity_type") in elasticity_filter)
        and (not product_filter or r.get("target_product") in product_filter)
        and (not food_group_filter or r.get("target_food_group") in food_group_filter)
    ]

    papers_total = len(list(INPUT_DIR.glob("*.pdf")))
    papers_with_records = len({r["paper_id"] for r in records if r.get("paper_id")})

    return jsonify({
        "papers_total": papers_total,
        "papers_with_records": papers_with_records,
        "chunks_total": len(_load_json(OUTPUT_DIR / "chunks.json", [])),
        "chunks_kept": len(kept),
        "chunks_classified": len(classifiable),
        "chunks_detected_positive": len(extractable),
        "records_total": len(records),
        "records_total_unfiltered": len(all_records),
        "records_requires_review": sum(1 for r in records if r.get("requires_review")),
        "records_manually_edited": sum(1 for r in records if r.get("manually_edited")),
        "products": _top_counts(records, "target_product"),
        "food_groups": _top_counts(records, "target_food_group"),
        "elasticity_types": _top_counts(records, "target_elasticity_type"),
        "countries": _top_counts(records, "countries_region"),
        "data_sources": _top_counts(records, "data_source"),
        "model_types": _top_counts(records, "model_type"),
        "specification_status": _top_counts(records, "specification_status"),
        "coefficient_summary": _numeric_summary(records, "coefficient"),
        "regression_runs": _load_regression_runs(),
    })


# --------------------------------------------------------------------------- #
# Thumbnails (first page of each PDF, rendered lazily and cached)
# --------------------------------------------------------------------------- #
def _clear_thumbnail_cache_for_missing_papers():
    existing_ids = {f.stem for f in INPUT_DIR.glob("*.pdf")}
    for thumb in THUMB_DIR.glob("*.png"):
        if thumb.stem not in existing_ids:
            thumb.unlink()


def _render_thumbnail(pdf_path: Path, out_path: Path) -> bool:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(0.6, 0.6))
        pix.save(str(out_path))
        return True
    except Exception:
        return False


@app.route("/api/papers/<paper_id>/thumbnail", methods=["GET"])
def paper_thumbnail(paper_id):
    thumb_path = THUMB_DIR / f"{paper_id}.png"
    if not thumb_path.exists():
        pdf_matches = list(INPUT_DIR.glob(f"{paper_id}.pdf"))
        if not pdf_matches:
            abort(404)
        ok = _render_thumbnail(pdf_matches[0], thumb_path)
        if not ok:
            abort(404)
    return send_file(thumb_path, mimetype="image/png")


if __name__ == "__main__":
    # threaded=True is required here: the classify endpoint blocks for the
    # whole run (one request per chunk to the LLM), so pause/resume/progress
    # requests need to be served concurrently on a separate thread rather
    # than queuing behind it.
    # host="0.0.0.0" is required for any non-local deployment (a droplet, a
    # VM, a container) — Flask's default host is 127.0.0.1, which only
    # accepts connections from inside the machine itself. Left unset, the
    # app runs fine over SSH-local testing but is completely unreachable
    # from a browser hitting the server's public IP.
    app.run(host="0.0.0.0", debug=True, port=5050, threaded=True)
