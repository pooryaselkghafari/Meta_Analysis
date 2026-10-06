#!/usr/bin/env python3
"""Detached entrypoint for durable analyze/classify/detect/extract jobs.

Spawned by ``stage_jobs.start_job`` via ``subprocess.Popen(..., start_new_session=True)``
so the job outlives the gunicorn/Flask worker that accepted the HTTP start
request. Progress is written to ``projects/<id>/output/_progress_<stage>.json``.

Usage:
    python webapp/job_runner.py analyze <project_id>
    python webapp/job_runner.py classify <project_id>
    python webapp/job_runner.py detect <project_id>
    python webapp/job_runner.py extract <project_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

_WEBAPP = Path(__file__).resolve().parent
_BASE = _WEBAPP.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_WEBAPP))

import stage_jobs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2 or args[0] not in stage_jobs.JOB_FUNCS:
        print(
            "usage: job_runner.py <analyze|classify|detect|extract> <project_id>",
            file=sys.stderr,
        )
        return 2
    stage, project_id = args[0], args[1]
    try:
        stage_jobs.JOB_FUNCS[stage](project_id)
    except Exception as e:
        # Last-resort: make sure progress isn't left spinning if the job
        # crashes before its own finally block can clear running.
        try:
            progress = stage_jobs.read_progress(project_id, stage)
            progress["running"] = False
            progress["error"] = str(e)
            stage_jobs.write_progress(project_id, stage, progress)
            stage_jobs._release_lock(project_id, stage)
        except Exception:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
