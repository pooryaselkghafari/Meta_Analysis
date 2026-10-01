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

import csv
import io
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

from flask import Flask, Response, jsonify, render_template, request, send_file, abort
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import Pipeline, PipelineConfig, LLMClient, settings_store  # noqa: E402
from meta_pipeline import prompts_store, AVAILABLE_MODELS, effort_levels_for  # noqa: E402
from meta_pipeline.models import ChunkType  # noqa: E402
import regression  # noqa: E402 — Stage 5: meta-regression over records.json
import stage_jobs  # noqa: E402 — durable classify/detect/extract subprocess jobs

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


@app.before_request
def _rebind_active_project():
    """Every request re-reads active_project.json and repoints path globals.

    Required under multi-worker gunicorn: ``set_base_dir`` only updates the
    worker that handled create/activate. Without this, Settings save can
    write ``ai_settings.json`` / ``prompts_settings.json`` under one path
    while Cheap/Main AI readiness on another worker reads a different
    (empty) file — keys look unset and a second Save from a blank UI wipe
    the real prompts.
    """
    _apply_active_project()


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
STAGE_ORDER = ["corpus", "cheap_ai", "detection_ai", "main_ai", "dashboard"]
# Human label + the page where each stage is actually run — used both for
# "you need X first" messaging on a locked page and for the link that gets
# you there.
STAGE_LABELS = {
    "corpus": "Corpus analysis",
    "cheap_ai": "Cheap AI — Classification",
    "detection_ai": "Detection AI — Detection",
    "main_ai": "Main AI — Extraction",
    "dashboard": "Dashboard — Meta Analysis Results",
}
STAGE_HREFS = {
    "corpus": "/", "cheap_ai": "/cheap-ai", "detection_ai": "/detection-ai",
    "main_ai": "/main-ai", "dashboard": "/dashboard",
}
# Which stage must be *_done before a given page is allowed to open at all.
# "/" (Corpus/upload) and "/settings" have no prerequisite. Dashboard gates
# on main_ai_done — Chen Tables 1–4 are built from extracted records.
PAGE_REQUIRES = {
    "results": "corpus", "cheap_ai": "corpus", "detection_ai": "cheap_ai",
    "main_ai": "detection_ai", "dashboard": "main_ai",
    # Legacy /regression URL still loads a deprecation notice (no gate).
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
    """Deprecated: generic OLS/WLS picker replaced by Chen Tables 1–4 on
    the Dashboard. Keep the URL so old bookmarks don't 404."""
    return render_template("regression_deprecated.html")


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
    # Any in-flight/paused AI job still points at the old corpus — stop them
    # so Main AI doesn't resume a stale larger run after papers were removed.
    stage_jobs.stop_all_jobs(
        _get_active_project_id(),
        reason="Cleared because a paper was removed from the corpus.",
    )
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
    """The fixed Chen et al. nine product groups (see FoodGroup in
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

    # Corpus rebuild invalidates every AI stage — kill paused/running jobs so
    # their progress files can't resurrect the old totals on later pages.
    stage_jobs.stop_all_jobs(
        _get_active_project_id(),
        reason="Cleared because the corpus was re-analyzed.",
    )

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


# Per-call token budget for the cheap-model Stage 2/3 calls
# (classify_chunk/detect_estimate) — see the max_tokens docstring on both in
# llm.py for why this needs to be generous at all: an effort/thinking-capable
# model on the cheap slot can burn the whole cap on invisible reasoning
# tokens before writing any visible answer, silently returning an empty,
# unparseable response.
#
# Scales UP with corpus size rather than down: a bigger upload is a bigger,
# higher-stakes run where a handful of silently truncated chunks buried among
# hundreds is far more annoying to notice and re-run than the same failure on
# a 3-paper test run — so it's worth spending a bit more per call to make
# truncation as close to impossible as practical, and cost is a secondary
# concern next to a run finishing cleanly. A small upload doesn't need as
# much headroom, so it stays cheaper by default.
_CHEAP_CALL_MAX_TOKENS_TIERS = (
    (5, 512),      # <=5 papers
    (20, 1024),    # 6-20 papers
    (None, 2048),  # 21+ papers
)


def _cheap_call_max_tokens() -> int:
    n_papers = len(list(INPUT_DIR.glob("*.pdf")))
    for threshold, tokens in _CHEAP_CALL_MAX_TOKENS_TIERS:
        if threshold is None or n_papers <= threshold:
            return tokens
    return _CHEAP_CALL_MAX_TOKENS_TIERS[-1][1]  # unreachable, but keeps this defensive


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
# Durable stage jobs (classify / detect / extract)
#
# Long-running AI loops no longer execute inside the HTTP request (that is
# what timed out behind nginx/gunicorn when models got slower). Instead:
#   POST  -> stage_jobs.start_job() claims a lock, spawns a detached
#            subprocess (job_runner.py), returns immediately
#   GET /progress -> stage_jobs progress file (shared across gunicorn
#            workers); also reconciles stale PIDs if the job crashed
#   frontend polls /progress until running=false — progress is the source
#            of truth, not the POST response
# See webapp/stage_jobs.py for the full contract.
# --------------------------------------------------------------------------- #


def _read_progress(stage: str, default: dict | None = None) -> dict:
    # `default` kept for call-site compatibility; stage_jobs owns the real
    # defaults and fills missing keys. Always reconcile stale PIDs so a
    # crashed job can't leave the UI spinning on running:true forever.
    return stage_jobs.reconcile_stale_job(_get_active_project_id(), stage)


def _write_progress(stage: str, data: dict) -> None:
    stage_jobs.write_progress(_get_active_project_id(), stage, data)


@app.route("/api/cheap-ai/classify/progress", methods=["GET"])
def cheap_ai_classify_progress():
    return jsonify(_read_progress("classify"))


@app.route("/api/cheap-ai/classify/pause", methods=["POST"])
def cheap_ai_classify_pause():
    progress = _read_progress("classify")
    if not progress["running"]:
        return jsonify({"error": "no classification run is currently in progress"}), 400
    progress["paused"] = True
    _write_progress("classify", progress)
    return jsonify(progress)


@app.route("/api/cheap-ai/classify/resume", methods=["POST"])
def cheap_ai_classify_resume():
    progress = _read_progress("classify")
    progress["paused"] = False
    _write_progress("classify", progress)
    return jsonify(progress)


@app.route("/api/cheap-ai/classify/stop", methods=["POST"])
def cheap_ai_classify_stop():
    progress = stage_jobs.stop_job(_get_active_project_id(), "classify", reason=None)
    progress["stopped"] = True
    return jsonify(progress)


@app.route("/api/detection-ai/detect/progress", methods=["GET"])
def detection_ai_detect_progress():
    return jsonify(_read_progress("detect"))


@app.route("/api/detection-ai/detect/pause", methods=["POST"])
def detection_ai_detect_pause():
    progress = _read_progress("detect")
    if not progress["running"]:
        return jsonify({"error": "no detection run is currently in progress"}), 400
    progress["paused"] = True
    _write_progress("detect", progress)
    return jsonify(progress)


@app.route("/api/detection-ai/detect/resume", methods=["POST"])
def detection_ai_detect_resume():
    progress = _read_progress("detect")
    progress["paused"] = False
    _write_progress("detect", progress)
    return jsonify(progress)


@app.route("/api/detection-ai/detect/stop", methods=["POST"])
def detection_ai_detect_stop():
    progress = stage_jobs.stop_job(_get_active_project_id(), "detect", reason=None)
    progress["stopped"] = True
    return jsonify(progress)


@app.route("/api/main-ai/extract/progress", methods=["GET"])
def main_ai_extract_progress():
    """Return extract progress, but auto-stop a stale paused/running job whose
    ``total`` no longer matches the current extractable corpus (e.g. papers
    removed + Cheap/Detection re-run while Main AI was still paused)."""
    project_id = _get_active_project_id()
    progress = stage_jobs.reconcile_stale_job(project_id, "extract")
    if progress.get("running"):
        extractable = _extractable_chunks(_kept_chunks())
        expected = len(extractable)
        reported = progress.get("total")
        if isinstance(reported, int) and reported > 0 and expected != reported:
            progress = stage_jobs.stop_job(
                project_id, "extract",
                reason="Cleared — extractable corpus changed since this run started. Click Run extraction to start on the current papers.",
            )
            progress["stopped"] = True
    return jsonify(progress)


@app.route("/api/main-ai/extract/pause", methods=["POST"])
def main_ai_extract_pause():
    progress = _read_progress("extract")
    if not progress["running"]:
        return jsonify({"error": "no extraction run is currently in progress"}), 400
    progress["paused"] = True
    _write_progress("extract", progress)
    return jsonify(progress)


@app.route("/api/main-ai/extract/resume", methods=["POST"])
def main_ai_extract_resume():
    progress = _read_progress("extract")
    progress["paused"] = False
    _write_progress("extract", progress)
    return jsonify(progress)


@app.route("/api/main-ai/extract/stop", methods=["POST"])
def main_ai_extract_stop():
    progress = stage_jobs.stop_job(_get_active_project_id(), "extract", reason=None)
    progress["stopped"] = True
    return jsonify(progress)


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
    chunk["classification_parse_error_raw"] = None
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
    """Start Stage 2 classification as a durable background job.

    Returns immediately with ``{started: true, total}``. The frontend must
    poll ``/api/cheap-ai/classify/progress`` until ``running`` is false —
    progress is the source of truth, not this response. Refuses (400) if the
    cheap AI's API key isn't set or the classification prompt is blank.
    """
    classifiable = _classifiable_chunks(_kept_chunks())
    if not classifiable:
        return jsonify({"error": "no classifiable kept chunks yet — run Analyze on the Corpus page first"}), 400

    err = _ai_readiness_error("cheap", "classification")
    if err:
        return jsonify({"error": err}), 400

    project_id = _get_active_project_id()
    try:
        stage_jobs.start_job(
            project_id, "classify",
            total=len(classifiable),
            paper_id=classifiable[0]["paper_id"],
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    return jsonify({"started": True, "total": len(classifiable)})


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
    doubles as the detected yes/no call — see
    LLMClient.detected_from_keep_reason / _DISCARD_KEEP_REASONS (off_target,
    no_estimate_signal, literature_estimate_only → no; every other reason →
    yes). Same rule the parser applies when reconciling a model response
    that contradicts itself on detected vs keep_reason. A manual "no" drops
    the chunk from the kept set the same way a cheap-LLM "no" does
    (passes_filter=False); a manual "yes" undoes that. Extraction records
    already built from this chunk are purged, scoped to just this chunk."""
    data = request.get_json(force=True) or {}
    keep_reason = data.get("keep_reason")
    if keep_reason not in LLMClient._KEEP_REASONS:
        return jsonify({"error": f"keep_reason must be one of {list(LLMClient._KEEP_REASONS)}"}), 400

    kept = _kept_chunks()
    chunk = _find_chunk(kept, chunk_id)
    if chunk is None or chunk.get("paper_id") != paper_id:
        return jsonify({"error": "chunk not found"}), 404

    detected = LLMClient.detected_from_keep_reason(keep_reason)
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
    """Start Stage 3 detection as a durable background job.

    Returns immediately with ``{started: true, total}``. Poll
    ``/api/detection-ai/detect/progress`` until ``running`` is false.
    Refuses (400) if the cheap AI's API key isn't set or the detection
    prompt is blank.
    """
    classifiable = _classifiable_chunks(_kept_chunks())
    if not classifiable:
        return jsonify({"error": "no classifiable kept chunks yet — run Analyze on the Corpus page first"}), 400

    err = _ai_readiness_error("cheap", "detection")
    if err:
        return jsonify({"error": err}), 400

    project_id = _get_active_project_id()
    try:
        stage_jobs.start_job(
            project_id, "detect",
            total=len(classifiable),
            paper_id=classifiable[0]["paper_id"],
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    return jsonify({"started": True, "total": len(classifiable)})


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


# Flat CSV columns for Main AI export — nested JSON fields are expanded so
# the file opens cleanly in Excel/Sheets without a second parse step.
_CSV_COLUMNS = [
    "estimate_id", "paper_id", "source_chunk_id",
    "target_elasticity_type", "target_product", "target_food_group",
    "target_cross_price_product", "target_cross_price_food_group",
    "paper_elasticity_wording_raw", "paper_product_wording_raw",
    "paper_cross_price_product_wording_raw",
    "match_type", "target_match_justification", "variable_role", "estimate_type",
    "coefficient", "standard_error", "standard_error_reported",
    "ci_low", "ci_high", "confidence_interval_reported",
    "p_value", "p_value_reported", "significance_stars",
    "test_statistic_type", "test_statistic_value",
    "elasticity_is_raw", "elasticity_transformation_type",
    "elasticity_unit", "unit_source",
    "specification_status", "baseline_evidence", "model_type",
    "countries_region", "time_period_start", "time_period_end", "frequency",
    "n_obs", "n_units", "data_source",
    "real_per_capita_income", "data_level", "geographic_scope", "urban_rural",
    "data_type", "price_measure", "conditioning", "budgeting_stages",
    "demand_system", "estimation_method", "demographic_controls",
    "publication_status", "study_language", "n_products_in_demand_system",
    "elasticity_form", "budget_share",
    "group_expenditure_elasticity", "group_own_price_elasticity",
    "within_group_budget_share",
    "source_type", "source_row", "source_column", "source_page",
    "table_complete", "pages_used",
    "requires_review", "review_reason",
    "manually_edited", "manually_added",
]


def _record_to_csv_row(r: dict) -> dict:
    ci = r.get("confidence_interval") or []
    tp = r.get("time_period") or {}
    ts = r.get("test_statistic") or {}
    loc = r.get("source_location") or {}
    reasons = r.get("review_reason") or []
    pages = r.get("pages_used") or []
    return {
        "estimate_id": r.get("estimate_id"),
        "paper_id": r.get("paper_id"),
        "source_chunk_id": r.get("source_chunk_id"),
        "target_elasticity_type": r.get("target_elasticity_type"),
        "target_product": r.get("target_product"),
        "target_food_group": r.get("target_food_group"),
        "target_cross_price_product": r.get("target_cross_price_product"),
        "target_cross_price_food_group": r.get("target_cross_price_food_group"),
        "paper_elasticity_wording_raw": r.get("paper_elasticity_wording_raw"),
        "paper_product_wording_raw": r.get("paper_product_wording_raw"),
        "paper_cross_price_product_wording_raw": r.get("paper_cross_price_product_wording_raw"),
        "match_type": r.get("match_type"),
        "target_match_justification": r.get("target_match_justification"),
        "variable_role": r.get("variable_role"),
        "estimate_type": r.get("estimate_type"),
        "coefficient": r.get("coefficient"),
        "standard_error": r.get("standard_error"),
        "standard_error_reported": r.get("standard_error_reported"),
        "ci_low": ci[0] if len(ci) > 0 else None,
        "ci_high": ci[1] if len(ci) > 1 else None,
        "confidence_interval_reported": r.get("confidence_interval_reported"),
        "p_value": r.get("p_value"),
        "p_value_reported": r.get("p_value_reported"),
        "significance_stars": r.get("significance_stars"),
        "test_statistic_type": ts.get("type") if isinstance(ts, dict) else None,
        "test_statistic_value": ts.get("value") if isinstance(ts, dict) else None,
        "elasticity_is_raw": r.get("elasticity_is_raw"),
        "elasticity_transformation_type": r.get("elasticity_transformation_type"),
        "elasticity_unit": r.get("elasticity_unit"),
        "unit_source": r.get("unit_source"),
        "specification_status": r.get("specification_status"),
        "baseline_evidence": r.get("baseline_evidence"),
        "model_type": r.get("model_type"),
        "countries_region": r.get("countries_region"),
        "time_period_start": tp.get("start") if isinstance(tp, dict) else None,
        "time_period_end": tp.get("end") if isinstance(tp, dict) else None,
        "frequency": r.get("frequency"),
        "n_obs": r.get("n_obs"),
        "n_units": r.get("n_units"),
        "data_source": r.get("data_source"),
        "real_per_capita_income": r.get("real_per_capita_income"),
        "data_level": r.get("data_level"),
        "geographic_scope": r.get("geographic_scope"),
        "urban_rural": r.get("urban_rural"),
        "data_type": r.get("data_type"),
        "price_measure": r.get("price_measure"),
        "conditioning": r.get("conditioning"),
        "budgeting_stages": r.get("budgeting_stages"),
        "demand_system": r.get("demand_system"),
        "estimation_method": r.get("estimation_method"),
        "demographic_controls": r.get("demographic_controls"),
        "publication_status": r.get("publication_status"),
        "study_language": r.get("study_language"),
        "n_products_in_demand_system": r.get("n_products_in_demand_system"),
        "elasticity_form": r.get("elasticity_form"),
        "budget_share": r.get("budget_share"),
        "group_expenditure_elasticity": r.get("group_expenditure_elasticity"),
        "group_own_price_elasticity": r.get("group_own_price_elasticity"),
        "within_group_budget_share": r.get("within_group_budget_share"),
        "source_type": r.get("source_type"),
        "source_row": loc.get("row") if isinstance(loc, dict) else None,
        "source_column": loc.get("column") if isinstance(loc, dict) else None,
        "source_page": loc.get("page") if isinstance(loc, dict) else None,
        "table_complete": r.get("table_complete"),
        "pages_used": ";".join(str(p) for p in pages) if pages else None,
        "requires_review": r.get("requires_review"),
        "review_reason": "; ".join(str(x) for x in reasons) if reasons else None,
        "manually_edited": r.get("manually_edited"),
        "manually_added": r.get("manually_added"),
    }


@app.route("/api/main-ai/records.csv", methods=["GET"])
def main_ai_records_csv():
    """Download extraction results as CSV. Honors the same filters as the
    Main AI table toolbar: ``?paper_id=`` and ``?review_only=1``."""
    records = _load_json(OUTPUT_DIR / "records.json", [])
    paper_id = (request.args.get("paper_id") or "").strip()
    review_only = request.args.get("review_only") in ("1", "true", "yes")
    if paper_id:
        records = [r for r in records if r.get("paper_id") == paper_id]
    if review_only:
        records = [r for r in records if r.get("requires_review")]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        writer.writerow(_record_to_csv_row(r))

    project = _get_active_project_id()
    filename = f"metaextract_{project}_records.csv"
    return Response(
        buf.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


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
    "data_level": LLMClient._DATA_LEVELS,
    "geographic_scope": LLMClient._GEOGRAPHIC_SCOPES,
    "urban_rural": LLMClient._URBAN_RURAL,
    "data_type": LLMClient._DATA_TYPES,
    "price_measure": LLMClient._PRICE_MEASURES,
    "conditioning": LLMClient._CONDITIONINGS,
    "budgeting_stages": LLMClient._BUDGETING_STAGES,
    "demand_system": LLMClient._DEMAND_SYSTEMS,
    "estimation_method": LLMClient._ESTIMATION_METHODS,
    "elasticity_form": LLMClient._ELASTICITY_FORMS,
    "publication_status": LLMClient._PUBLICATION_STATUSES,
    "study_language": LLMClient._STUDY_LANGUAGES,
}
_RECORD_NUMERIC_FIELDS = {
    "coefficient", "standard_error", "p_value", "n_obs", "n_units",
    "real_per_capita_income", "budget_share",
    "group_expenditure_elasticity", "group_own_price_elasticity",
    "within_group_budget_share", "n_products_in_demand_system",
}
_RECORD_INT_FIELDS = {"n_obs", "n_units", "n_products_in_demand_system"}
_RECORD_BOOL_FIELDS = {
    "elasticity_is_raw", "requires_review", "demographic_controls",
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
        "real_per_capita_income": None, "data_level": None, "geographic_scope": None,
        "urban_rural": None, "data_type": None, "price_measure": None,
        "conditioning": None, "budgeting_stages": None, "demand_system": None,
        "estimation_method": None, "demographic_controls": None,
        "publication_status": None, "study_language": None,
        "n_products_in_demand_system": None, "elasticity_form": None,
        "budget_share": None, "group_expenditure_elasticity": None,
        "group_own_price_elasticity": None, "within_group_budget_share": None,
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
    # Recompute Chen missing-data review flags after edits (e.g. filling income
    # or budget share should clear those auto-reasons).
    LLMClient._apply_chen_review_flags(rec)
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
    for field in ("model_type", "countries_region", "frequency", "n_obs", "n_units", "data_source",
                  *LLMClient._CHEN_PAPER_FIELDS):
        if rec.get(field) is None:
            rec[field] = meta.get(field)
    tp = rec.get("time_period")
    if not tp or (tp.get("start") is None and tp.get("end") is None):
        rec["time_period"] = meta.get("time_period") or {"start": None, "end": None}
    LLMClient._apply_chen_review_flags(rec)
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
    """Start Stage 4 extraction as a durable background job.

    Returns immediately with ``{started: true, total}``. Poll
    ``/api/main-ai/extract/progress`` until ``running`` is false. Paper
    metadata (Stage 4a) runs inside the job process — not this request —
    so a slow main model can't time out the start POST. Refuses (400) if
    the main AI's API key isn't set or the extraction prompt is blank.
    """
    extractable = _extractable_chunks(_kept_chunks())
    if not extractable:
        return jsonify({"error": "no chunks have passed detection yet — run Detection AI first"}), 400

    err = _ai_readiness_error("main", "extraction")
    if err:
        return jsonify({"error": err}), 400

    project_id = _get_active_project_id()
    try:
        stage_jobs.start_job(
            project_id, "extract",
            total=len(extractable),
            paper_id=extractable[0]["paper_id"],
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    return jsonify({"started": True, "total": len(extractable)})


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
# Dashboard — Chen et al. Tables 1–4 over extracted records (country filter
# refilters the sample and recomputes descriptives + meta-regression).
# --------------------------------------------------------------------------- #
def _multi_query_param(name: str) -> list:
    """Reads a filter param as repeated keys (?country=a&country=b) or one
    comma-separated value. Empty/blank entries dropped; empty list = no filter."""
    out = []
    for raw in request.args.getlist(name):
        out.extend(v.strip() for v in raw.split(",") if v.strip())
    return out


@app.route("/api/dashboard/summary", methods=["GET"])
def dashboard_summary():
    """Build Chen-style Tables 1–4 from records.json. Optional repeated
    ``?country=`` query params restrict the sample (exact match on
    ``countries_region``)."""
    import chen_meta

    all_records = _load_json(OUTPUT_DIR / "records.json", [])
    countries = _multi_query_param("country")
    try:
        payload = chen_meta.build_dashboard(all_records, countries or None)
    except Exception as e:
        return jsonify({"error": f"Dashboard build failed: {e}"}), 500
    return jsonify(payload)


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
    # host="0.0.0.0" is required for any non-local deployment (a droplet, a
    # VM, a container) — Flask's default host is 127.0.0.1, which only
    # accepts connections from inside the machine itself. Left unset, the
    # app runs fine over SSH-local testing but is completely unreachable
    # from a browser hitting the server's public IP.
    # threaded=True still helps for concurrent progress polls / pause clicks
    # while other short requests are in flight; the long AI stages themselves
    # now run in detached subprocesses (see stage_jobs), not in request threads.
    app.run(host="0.0.0.0", debug=True, port=5050, threaded=True)
