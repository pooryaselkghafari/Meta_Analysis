"""Persistence + placeholder-filling for the four LLM prompts:
Stage 2 classification, Stage 3 detection, Stage 4 extraction, Stage 5
validation.

Stored as plain JSON at the project root (`prompts_settings.json`), edited
from the Settings page. `prompts.py` reads through this module, so the actual
prompt text lives in one editable place rather than hardcoded in Python —
that's the whole point of exposing it in the UI.

Placeholders use a literal {{token}} syntax rather than str.format, on
purpose: prompt authors will often paste literal JSON (schema examples,
sample output) into a template, and str.format would choke on every stray `{`
in that JSON unless it was escaped as `{{`. A plain string-replace on
`{{token}}` avoids that entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default location before any project has been activated; repointed at the
# active project's own directory via set_base_dir(), same reasoning as
# settings_store — each project keeps its own prompt templates.
PROMPTS_PATH = PROJECT_ROOT / "prompts_settings.json"

_STAGES = ("classification", "detection", "paper_metadata", "extraction", "validation")


def set_base_dir(base_dir) -> None:
    """Repoint PROMPTS_PATH at `<base_dir>/prompts_settings.json` — called by
    the webapp when the active project changes."""
    global PROMPTS_PATH
    PROMPTS_PATH = Path(base_dir) / "prompts_settings.json"

# Which {{tokens}} each stage's templates can reference — surfaced to the
# Settings page so prompt authors know what's available without reading code.
TOKENS: Dict[str, list] = {
    "classification": ["chunk_text"],
    "detection": ["chunk_text", "targets", "classification"],
    "paper_metadata": ["abstract", "methodology_text"],
    "extraction": ["chunk_text", "targets", "ontology", "abstract", "screening"],
    "validation": ["record_json", "chunk_text"],
}


def _read_raw() -> Dict[str, Any]:
    if PROMPTS_PATH.exists():
        try:
            return json.loads(PROMPTS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load() -> Dict[str, Dict[str, str]]:
    """All four stages' saved (system, user) templates, defaulting to blank."""
    data = _read_raw()
    return {
        stage: {
            "system": (data.get(stage) or {}).get("system", "") or "",
            "user": (data.get(stage) or {}).get("user", "") or "",
        }
        for stage in _STAGES
    }


def save(payload: Dict[str, Any]) -> None:
    """Update one or more stages' templates. `payload` is
    {"classification": {"system": ..., "user": ...}, ...} for any subset of
    stages; omitted stages are left untouched."""
    existing = _read_raw()
    for stage in _STAGES:
        if stage not in payload:
            continue
        existing.setdefault(stage, {})
        incoming = payload[stage] or {}
        if "system" in incoming:
            existing[stage]["system"] = incoming["system"] or ""
        if "user" in incoming:
            existing[stage]["user"] = incoming["user"] or ""
    PROMPTS_PATH.write_text(json.dumps(existing, indent=2))


def fill(template: str, **kwargs: Any) -> str:
    """Replace every {{key}} in `template` with its value. Non-string values
    (dicts, lists, None) are JSON-encoded first."""
    out = template
    for key, value in kwargs.items():
        token = "{{" + key + "}}"
        if token not in out:
            continue
        if value is None:
            text = ""
        elif isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, indent=2, default=str)
        out = out.replace(token, text)
    return out


def render(stage: str, **kwargs: Any) -> Tuple[str, str]:
    """Return the (system, user) prompt for a stage with placeholders filled.

    Returns ("", "") untouched — no filling attempted — when neither field has
    been authored yet, so LLMClient's existing blank-prompt short-circuit
    (skip the API call, return None) keeps working exactly as it did before
    prompts became UI-editable.
    """
    saved = load().get(stage, {"system": "", "user": ""})
    system, user = saved.get("system", ""), saved.get("user", "")
    if not system.strip() and not user.strip():
        return "", ""
    return fill(system, **kwargs), fill(user, **kwargs)
