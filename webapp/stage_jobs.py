"""Durable stage jobs for classify / detect / extract.

Each long-running AI stage runs in a **detached subprocess** (see
``job_runner.py`` + ``start_job``), not inside a gunicorn/Flask worker
request or a daemon thread. The HTTP layer only:

  1. checks readiness,
  2. claims the job (file lock),
  3. spawns the subprocess,
  4. returns immediately.

Progress is the shared source of truth: a small JSON file under the
project's ``output/`` directory that every worker (and the job process)
can read/write. The frontend polls that file until ``running`` is false.

This survives nginx/gunicorn request timeouts and model latency changes —
those only affect how long the job process runs, not whether the HTTP
request stays open.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Repo root on path so meta_pipeline imports work whether we're launched
# from Flask (webapp/ on path) or as job_runner.py (repo root on path).
_WEBAPP_DIR = Path(__file__).resolve().parent
_BASE_DIR = _WEBAPP_DIR.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))
if str(_WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(_WEBAPP_DIR))

from meta_pipeline import LLMClient, prompts_store, settings_store  # noqa: E402
from meta_pipeline.models import ChunkType  # noqa: E402

PROJECTS_DIR = _BASE_DIR / "projects"
JOB_RUNNER = _WEBAPP_DIR / "job_runner.py"

STAGES = ("classify", "detect", "extract")

# When an earlier stage is re-started, these later jobs (and their paused
# progress files) are obsolete and must be stopped/cleared.
DOWNSTREAM_STAGES: Dict[str, Tuple[str, ...]] = {
    "classify": ("detect", "extract"),
    "detect": ("extract",),
    "extract": (),
}

PROGRESS_DEFAULTS: Dict[str, dict] = {
    "classify": {
        "running": False,
        "paused": False,
        "stop_requested": False,
        "stopped": False,
        "pid": None,
        "paper_id": None,
        "chunk_index": 0,
        "total": 0,
        "classified": 0,
        "parse_failed": 0,
        "completed": 0,
        "error": None,
    },
    "detect": {
        "running": False,
        "paused": False,
        "stop_requested": False,
        "stopped": False,
        "pid": None,
        "paper_id": None,
        "chunk_index": 0,
        "total": 0,
        "screened": 0,
        "dropped": 0,
        "completed": 0,
        "error": None,
    },
    "extract": {
        "running": False,
        "paused": False,
        "stop_requested": False,
        "stopped": False,
        "pid": None,
        "paper_id": None,
        "chunk_index": 0,
        "total": 0,
        "extracted": 0,
        "records_found": 0,
        "completed": 0,
        "duplicates_removed": 0,
        "error": None,
    },
}

_CHEAP_CALL_MAX_TOKENS_TIERS = (
    (5, 512),
    (20, 1024),
    (None, 2048),
)


# --------------------------------------------------------------------------- #
# Paths / IO
# --------------------------------------------------------------------------- #
def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def output_dir_for(project_id: str) -> Path:
    return project_dir(project_id) / "output"


def _bind_project_stores(project_id: str) -> Path:
    """Ensure the project's output dir exists and return it.

    API keys / prompts are global at the repo root — do not rebind them to
    the project directory (settings_store / prompts_store ignore project
    paths by design).
    """
    proj = project_dir(project_id)
    if not proj.is_dir():
        raise FileNotFoundError(f"project not found: {project_id}")
    out = proj / "output"
    out.mkdir(parents=True, exist_ok=True)
    # Pin stores to the shared root (no-op if already there).
    settings_store.set_base_dir(_BASE_DIR)
    prompts_store.set_base_dir(_BASE_DIR)
    return out


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2))


def progress_path(project_id: str, stage: str) -> Path:
    return output_dir_for(project_id) / f"_progress_{stage}.json"


def lock_path(project_id: str, stage: str) -> Path:
    return output_dir_for(project_id) / f"_job_{stage}.lock"


def log_path(project_id: str, stage: str) -> Path:
    return output_dir_for(project_id) / f"_job_{stage}.log"


def read_progress(project_id: str, stage: str) -> dict:
    default = PROGRESS_DEFAULTS[stage]
    path = progress_path(project_id, stage)
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return dict(default)
    # Fill any missing keys from defaults so older progress files stay valid.
    out = dict(default)
    out.update(data)
    return out


def write_progress(project_id: str, stage: str, data: dict) -> None:
    path = progress_path(project_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


class JobStopped(Exception):
    """Raised inside a job loop when the user (or an upstream re-run) asked
    the stage to stop. Callers catch it to exit cleanly before ``finally``."""


def _wait_if_paused(project_id: str, stage: str, progress: dict,
                    on_pause: Optional[Callable[[], None]] = None) -> None:
    """Block between units of work while progress.paused is true.

    Pause/resume is file-backed (``/pause`` and ``/resume`` flip the flag from
    any gunicorn worker). ``/stop`` sets ``stop_requested`` and clears
    ``paused`` so a paused job wakes and exits. Optional ``on_pause`` persists
    intermediate results so a long pause doesn't leave unflushed work only in
    memory.
    """
    while True:
        live = read_progress(project_id, stage)
        if live.get("stop_requested"):
            progress["stop_requested"] = True
            progress["paused"] = False
            raise JobStopped()
        if not live.get("paused"):
            break
        progress["paused"] = True
        write_progress(project_id, stage, progress)
        if on_pause:
            on_pause()
        time.sleep(0.5)
    progress["paused"] = False


def _job_pids(project_id: str, stage: str) -> List[int]:
    """PIDs that might still own this stage job (progress pid and/or lock)."""
    pids: List[int] = []
    progress = read_progress(project_id, stage)
    pid = progress.get("pid")
    if isinstance(pid, int) and pid > 0:
        pids.append(pid)
    lock = lock_path(project_id, stage)
    if lock.exists():
        try:
            lock_pid = int(lock.read_text().strip() or "0")
        except ValueError:
            lock_pid = 0
        if lock_pid > 0 and lock_pid not in pids:
            pids.append(lock_pid)
    return pids


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop_job(project_id: str, stage: str, *, reason: Optional[str] = None) -> dict:
    """Stop a running/paused stage job and reset its progress file.

    Sets ``stop_requested`` (and clears ``paused``) so a cooperative job can
    exit its loop, then SIGTERM/SIGKILL if the process is still alive. Always
    leaves ``running=False`` so the UI doesn't keep showing a stale pause
    from a superseded corpus.
    """
    if stage not in PROGRESS_DEFAULTS:
        raise ValueError(f"unknown stage: {stage}")

    progress = read_progress(project_id, stage)
    progress["stop_requested"] = True
    progress["paused"] = False
    write_progress(project_id, stage, progress)

    pids = _job_pids(project_id, stage)
    # Give cooperative exit a moment (pause wake + loop check).
    deadline = time.time() + 8.0
    while time.time() < deadline and any(pid_alive(p) for p in pids):
        time.sleep(0.2)
        pids = _job_pids(project_id, stage)

    for p in pids:
        if pid_alive(p):
            _signal_pid(p, signal.SIGTERM)
    time.sleep(0.4)
    for p in _job_pids(project_id, stage):
        if pid_alive(p):
            _signal_pid(p, signal.SIGKILL)

    cleared = dict(PROGRESS_DEFAULTS[stage])
    cleared["stopped"] = True
    if reason:
        cleared["error"] = reason
    write_progress(project_id, stage, cleared)
    _release_lock(project_id, stage)
    return cleared


def invalidate_downstream_jobs(project_id: str, stage: str,
                               *, reason: Optional[str] = None) -> None:
    """Kill/clear later-stage jobs whose inputs this stage just invalidated.

    Fixes the stuck-pause case: Main AI paused on corpus A, user re-runs
    Cheap/Detection AI on a smaller corpus — without this, extract progress
    stays ``running+paused`` with the old totals.
    """
    msg = reason or "Cleared because an earlier pipeline stage was re-run."
    for ds in DOWNSTREAM_STAGES.get(stage, ()):
        stop_job(project_id, ds, reason=msg)


def stop_all_jobs(project_id: str, *, reason: Optional[str] = None) -> None:
    msg = reason or "Cleared because the corpus changed."
    for stage in STAGES:
        stop_job(project_id, stage, reason=msg)


def pid_alive(pid: Optional[int]) -> bool:
    if not pid or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, but not ours to signal — still running
    except OSError:
        return False
    return True


def reconcile_stale_job(project_id: str, stage: str) -> dict:
    """If progress says running but the job PID is dead, mark the job finished
    with an error so the UI doesn't spin forever after a crash/OOM/kill."""
    progress = read_progress(project_id, stage)
    if not progress.get("running"):
        return progress
    pid = progress.get("pid")
    if pid is None:
        # Claimed but subprocess hasn't written its pid yet — give it a
        # moment; only treat as stale if the lock is also gone/dead.
        lock = lock_path(project_id, stage)
        if lock.exists():
            try:
                lock_pid = int(lock.read_text().strip() or "0")
            except ValueError:
                lock_pid = 0
            if pid_alive(lock_pid):
                return progress
        progress["running"] = False
        progress["error"] = progress.get("error") or "job failed to start"
        write_progress(project_id, stage, progress)
        _release_lock(project_id, stage)
        return progress
    if pid_alive(pid):
        return progress
    progress["running"] = False
    progress["error"] = progress.get("error") or "job process exited unexpectedly"
    write_progress(project_id, stage, progress)
    _release_lock(project_id, stage)
    return progress


def job_is_running(project_id: str, stage: str) -> bool:
    return bool(reconcile_stale_job(project_id, stage).get("running"))


def _claim_lock(project_id: str, stage: str) -> bool:
    """Atomic claim via O_EXCL lock file. Returns False if another live job
    already holds the lock."""
    path = lock_path(project_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = int(path.read_text().strip() or "0")
        except ValueError:
            existing = 0
        if pid_alive(existing):
            return False
        try:
            path.unlink()
        except OSError:
            pass
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    return True


def _release_lock(project_id: str, stage: str) -> None:
    path = lock_path(project_id, stage)
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        # Python <3.8 compatibility — this deploy is 3.x modern, but be safe
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _adopt_lock(project_id: str, stage: str) -> None:
    """Job process: rewrite the lock file with this process's PID."""
    path = lock_path(project_id, stage)
    path.write_text(str(os.getpid()))


# --------------------------------------------------------------------------- #
# Chunk helpers (mirrored from app.py so the job process is self-contained)
# --------------------------------------------------------------------------- #
def _kept_chunks(out: Path) -> list:
    return _load_json(out / "chunks_kept.json", [])


def _save_kept_chunks(out: Path, chunks: list) -> None:
    _write_json(out / "chunks_kept.json", chunks)


def _classifiable_chunks(chunks: list) -> list:
    return [
        c for c in chunks
        if not c.get("is_abstract") and c.get("passes_filter", True) is not False
    ]


def _extractable_chunks(chunks: list) -> list:
    return [c for c in _classifiable_chunks(chunks) if c.get("detected") is True]


def _cheap_call_max_tokens(input_dir: Path) -> int:
    n_papers = len(list(input_dir.glob("*.pdf")))
    for threshold, tokens in _CHEAP_CALL_MAX_TOKENS_TIERS:
        if threshold is None or n_papers <= threshold:
            return tokens
    return _CHEAP_CALL_MAX_TOKENS_TIERS[-1][1]


def _discard_extraction_results(out: Path) -> None:
    chunks = _kept_chunks(out)
    changed = False
    for c in chunks:
        if c.get("extraction_done") is not None:
            c["extraction_done"] = None
            changed = True
    if changed:
        _save_kept_chunks(out, chunks)
    _write_json(out / "records.json", [])


def _discard_detection_results(out: Path) -> None:
    chunks = _kept_chunks(out)
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
        _save_kept_chunks(out, chunks)
    _discard_extraction_results(out)


def _abstracts_by_paper(out: Path) -> dict:
    abstracts = _load_json(out / "abstracts.json", [])
    return {a["paper_id"]: a.get("abstract") for a in abstracts}


def _methodology_text_by_paper(chunks: list) -> dict:
    by_paper: dict = {}
    for c in chunks:
        labels = c.get("labels") or ([c["chunk_type"]] if c.get("chunk_type") else [])
        if "methodology" in labels or "data_description" in labels:
            by_paper.setdefault(c["paper_id"], []).append(c["text"])
    return {pid: "\n\n".join(texts)[:12000] for pid, texts in by_paper.items()}


def _record_from_estimate(raw: dict, chunk: dict, idx: int, meta: dict) -> dict:
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
    if not tp or (isinstance(tp, dict) and tp.get("start") is None and tp.get("end") is None):
        rec["time_period"] = meta.get("time_period") or {"start": None, "end": None}
    LLMClient._apply_chen_review_flags(rec)
    return rec


def _decimal_places(v: float) -> int:
    s = f"{v:.10f}".rstrip("0")
    return len(s.split(".")[1]) if "." in s else 0


def _is_rounded_duplicate(rounded, raw) -> bool:
    if not isinstance(rounded, (int, float)) or not isinstance(raw, (int, float)):
        return False
    d_rounded, d_raw = _decimal_places(rounded), _decimal_places(raw)
    if d_rounded >= d_raw:
        return False
    return abs(round(raw, d_rounded) - rounded) < 1e-9


def _dedupe_rounded_records(records: list):
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


# --------------------------------------------------------------------------- #
# Job implementations
# --------------------------------------------------------------------------- #
def run_classify_job(project_id: str) -> None:
    out = _bind_project_stores(project_id)
    _adopt_lock(project_id, "classify")
    proj = project_dir(project_id)
    input_dir = proj / "input_papers"

    all_chunks = _kept_chunks(out)
    classifiable = _classifiable_chunks(all_chunks)
    total = len(classifiable)
    progress = dict(PROGRESS_DEFAULTS["classify"])
    progress.update(
        running=True, paused=False, pid=os.getpid(),
        paper_id=classifiable[0]["paper_id"] if classifiable else None,
        chunk_index=0, total=total, classified=0, parse_failed=0, completed=0, error=None,
    )
    write_progress(project_id, "classify", progress)

    if not classifiable:
        progress["running"] = False
        progress["error"] = "no classifiable kept chunks"
        write_progress(project_id, "classify", progress)
        _release_lock(project_id, "classify")
        return

    cfg = settings_store.load_model_config()
    llm = LLMClient(cfg)
    cheap_max_tokens = _cheap_call_max_tokens(input_dir)
    SAVE_EVERY = 5
    classified = 0
    parse_failures = 0

    try:
        for i, c in enumerate(classifiable):
            # Pause between chunks (file-backed, so /pause from any worker works).
            _wait_if_paused(
                project_id, "classify", progress,
                on_pause=lambda: _save_kept_chunks(out, all_chunks),
            )
            progress["paper_id"] = c["paper_id"]
            progress["chunk_index"] = i + 1
            write_progress(project_id, "classify", progress)

            try:
                result = llm.classify_chunk(c["text"], max_tokens=cheap_max_tokens)
            except Exception as e:
                _save_kept_chunks(out, all_chunks)
                progress["error"] = str(e)
                progress["running"] = False
                write_progress(project_id, "classify", progress)
                return

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
                c["classification_parse_error_raw"] = None
                classified += 1
            else:
                c["classification_parse_failed"] = True
                raw = getattr(llm, "last_classify_raw", None)
                c["classification_parse_error_raw"] = raw[:500] if raw else "(empty response)"
                parse_failures += 1

            progress["classified"] = classified
            progress["parse_failed"] = parse_failures
            progress["completed"] = i + 1
            if (i + 1) % SAVE_EVERY == 0:
                _save_kept_chunks(out, all_chunks)
            write_progress(project_id, "classify", progress)

        _save_kept_chunks(out, all_chunks)
        _discard_detection_results(out)
    except JobStopped:
        progress["stopped"] = True
        _save_kept_chunks(out, all_chunks)
    finally:
        progress["running"] = False
        progress["paused"] = False
        progress["stop_requested"] = False
        write_progress(project_id, "classify", progress)
        _release_lock(project_id, "classify")


def run_detect_job(project_id: str) -> None:
    out = _bind_project_stores(project_id)
    _adopt_lock(project_id, "detect")
    proj = project_dir(project_id)
    input_dir = proj / "input_papers"
    targets_path = proj / "targets.json"

    all_chunks = _kept_chunks(out)
    classifiable = _classifiable_chunks(all_chunks)
    total = len(classifiable)
    progress = dict(PROGRESS_DEFAULTS["detect"])
    progress.update(
        running=True, paused=False, pid=os.getpid(),
        paper_id=classifiable[0]["paper_id"] if classifiable else None,
        chunk_index=0, total=total, screened=0, dropped=0, completed=0, error=None,
    )
    write_progress(project_id, "detect", progress)

    if not classifiable:
        progress["running"] = False
        progress["error"] = "no classifiable kept chunks"
        write_progress(project_id, "detect", progress)
        _release_lock(project_id, "detect")
        return

    targets = _load_json(targets_path, {})
    cfg = settings_store.load_model_config()
    llm = LLMClient(cfg)
    cheap_max_tokens = _cheap_call_max_tokens(input_dir)
    SAVE_EVERY = 5
    screened = 0
    dropped = 0

    try:
        for i, c in enumerate(classifiable):
            _wait_if_paused(
                project_id, "detect", progress,
                on_pause=lambda: _save_kept_chunks(out, all_chunks),
            )
            progress["paper_id"] = c["paper_id"]
            progress["chunk_index"] = i + 1
            write_progress(project_id, "detect", progress)

            classification_context = {
                "labels": c.get("labels") or ([c["chunk_type"]] if c.get("chunk_type") else []),
                "primary": c.get("chunk_type"),
                "confidence": c.get("classification_confidence"),
                "keep": c.get("keep_classification"),
            }
            try:
                result = llm.detect_estimate(
                    c["text"], targets, classification_context, max_tokens=cheap_max_tokens,
                )
            except Exception as e:
                _save_kept_chunks(out, all_chunks)
                progress["error"] = str(e)
                progress["running"] = False
                write_progress(project_id, "detect", progress)
                return

            if result is not None:
                detected = result.get("detected")
                c["detected"] = detected
                c["detection_confidence"] = result.get("confidence")
                c["target_match"] = result.get("target_match")
                c["keep_reason"] = result.get("keep_reason")
                screened += 1
                if detected is False:
                    c["passes_filter"] = False
                    c["filter_reason"] = "cheap_llm_negative"
                    dropped += 1

            progress["screened"] = screened
            progress["dropped"] = dropped
            progress["completed"] = i + 1
            if (i + 1) % SAVE_EVERY == 0:
                _save_kept_chunks(out, all_chunks)
            write_progress(project_id, "detect", progress)

        _save_kept_chunks(out, all_chunks)
        _discard_extraction_results(out)
    except JobStopped:
        progress["stopped"] = True
        _save_kept_chunks(out, all_chunks)
    finally:
        progress["running"] = False
        progress["paused"] = False
        progress["stop_requested"] = False
        write_progress(project_id, "detect", progress)
        _release_lock(project_id, "detect")


def run_extract_job(project_id: str) -> None:
    out = _bind_project_stores(project_id)
    _adopt_lock(project_id, "extract")
    proj = project_dir(project_id)

    all_chunks = _kept_chunks(out)
    extractable = _extractable_chunks(all_chunks)
    total = len(extractable)
    records = _load_json(out / "records.json", [])
    progress = dict(PROGRESS_DEFAULTS["extract"])
    progress.update(
        running=True, paused=False, stop_requested=False, stopped=False,
        pid=os.getpid(),
        paper_id=extractable[0]["paper_id"] if extractable else None,
        chunk_index=0, total=total, extracted=0, records_found=len(records),
        completed=0, duplicates_removed=0, error=None,
    )
    write_progress(project_id, "extract", progress)

    if not extractable:
        progress["running"] = False
        progress["error"] = "no chunks have passed detection yet"
        write_progress(project_id, "extract", progress)
        _release_lock(project_id, "extract")
        return

    targets = _load_json(proj / "targets.json", {})
    ontology = _load_json(proj / "ontology.json", {})
    cfg = settings_store.load_model_config()
    llm = LLMClient(cfg)

    # Stage 4a — paper metadata runs INSIDE the job (not the HTTP start
    # request), so a slow main model + many papers can't time out the POST.
    abstracts = _abstracts_by_paper(out)
    methodology_text = _methodology_text_by_paper(all_chunks)
    paper_metadata_prompt = prompts_store.load().get("paper_metadata", {})
    paper_metadata_configured = bool(
        paper_metadata_prompt.get("system", "").strip()
        or paper_metadata_prompt.get("user", "").strip()
    )
    paper_metadata: dict = {}
    SAVE_EVERY = 3
    extracted_chunks = 0

    def _flush_extract():
        _save_kept_chunks(out, all_chunks)
        _write_json(out / "records.json", records)

    try:
        if paper_metadata_configured:
            paper_ids = sorted({c["paper_id"] for c in extractable})
            progress["paper_id"] = "paper metadata…"
            write_progress(project_id, "extract", progress)
            for paper_id in paper_ids:
                _wait_if_paused(project_id, "extract", progress)
                progress["paper_id"] = f"paper metadata… {paper_id}"
                write_progress(project_id, "extract", progress)
                try:
                    meta = llm.extract_paper_metadata(
                        abstracts.get(paper_id), methodology_text.get(paper_id, ""),
                    )
                except Exception:
                    meta = None
                if meta:
                    paper_metadata[paper_id] = meta

        for i, c in enumerate(extractable):
            _wait_if_paused(project_id, "extract", progress, on_pause=_flush_extract)
            progress["paper_id"] = c["paper_id"]
            progress["chunk_index"] = i + 1
            write_progress(project_id, "extract", progress)

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
                _save_kept_chunks(out, all_chunks)
                _write_json(out / "records.json", records)
                progress["error"] = str(e)
                progress["running"] = False
                write_progress(project_id, "extract", progress)
                return

            records = [r for r in records if r.get("source_chunk_id") != c["chunk_id"]]
            for idx, raw in enumerate(raw_estimates or []):
                records.append(_record_from_estimate(raw, c, idx, paper_metadata.get(c["paper_id"])))

            c["extraction_done"] = True
            extracted_chunks += 1
            progress["extracted"] = extracted_chunks
            progress["records_found"] = len(records)
            progress["completed"] = i + 1
            if (i + 1) % SAVE_EVERY == 0:
                _save_kept_chunks(out, all_chunks)
                _write_json(out / "records.json", records)
            write_progress(project_id, "extract", progress)

        records, deduped_count = _dedupe_rounded_records(records)
        _save_kept_chunks(out, all_chunks)
        _write_json(out / "records.json", records)
        progress["duplicates_removed"] = deduped_count
        progress["records_found"] = len(records)
    except JobStopped:
        progress["stopped"] = True
        _save_kept_chunks(out, all_chunks)
        _write_json(out / "records.json", records)
    finally:
        progress["running"] = False
        progress["paused"] = False
        progress["stop_requested"] = False
        write_progress(project_id, "extract", progress)
        _release_lock(project_id, "extract")


JOB_FUNCS: Dict[str, Callable[[str], None]] = {
    "classify": run_classify_job,
    "detect": run_detect_job,
    "extract": run_extract_job,
}


def start_job(project_id: str, stage: str, total: int, paper_id: Optional[str] = None) -> dict:
    """Claim the stage lock, write starting progress, spawn a detached
    subprocess. Returns the starting progress dict. Raises RuntimeError if
    a live job is already running for this stage/project."""
    if stage not in JOB_FUNCS:
        raise ValueError(f"unknown stage: {stage}")
    # Later stages (and any paused Main AI run) are obsolete once this stage
    # is re-started on a new/changed corpus.
    invalidate_downstream_jobs(project_id, stage)
    reconcile_stale_job(project_id, stage)
    if job_is_running(project_id, stage):
        raise RuntimeError(f"an {stage} run is already in progress")
    if not _claim_lock(project_id, stage):
        raise RuntimeError(f"an {stage} run is already in progress")

    progress = dict(PROGRESS_DEFAULTS[stage])
    progress.update(
        running=True, pid=None, paper_id=paper_id,
        chunk_index=0, total=total, completed=0, error=None,
        paused=False, stop_requested=False, stopped=False,
    )
    if stage == "classify":
        progress.update(classified=0, parse_failed=0)
    elif stage == "detect":
        progress.update(screened=0, dropped=0)
    else:
        progress.update(extracted=0, records_found=0, duplicates_removed=0)
    write_progress(project_id, stage, progress)

    log = log_path(project_id, stage)
    try:
        log_f = open(log, "a", buffering=1)
        log_f.write(f"\n--- starting {stage} job for project={project_id} pid_launcher={os.getpid()} ---\n")
        # start_new_session=True detaches from the gunicorn/Flask worker so
        # the job keeps running if that worker is recycled or the request
        # handler exits. Progress + lock files are how the webapp tracks it.
        proc = subprocess.Popen(
            [sys.executable, str(JOB_RUNNER), stage, project_id],
            cwd=str(_BASE_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
        # Record the child PID immediately (don't wait for the job to
        # adopt the lock itself). Otherwise a crash during job_runner
        # import leaves progress.running=True with pid=None and a lock
        # holding the still-alive gunicorn worker PID — reconcile would
        # think the job is "still starting" forever.
        progress["pid"] = proc.pid
        write_progress(project_id, stage, progress)
        lock_path(project_id, stage).write_text(str(proc.pid))
    except Exception:
        progress["running"] = False
        progress["error"] = "failed to spawn job process"
        write_progress(project_id, stage, progress)
        _release_lock(project_id, stage)
        raise

    return progress
