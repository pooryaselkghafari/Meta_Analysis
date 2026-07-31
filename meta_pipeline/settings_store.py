"""Persistence for the three AI slots (cheap / main / validation) configured
from the webapp's Settings page.

Stored as plain JSON at the project root (`ai_settings.json`, next to
`input_papers/`), *not* inside the `meta_pipeline` package, so it's easy to
gitignore and easy to find. This file will contain API keys in plain text —
that's an accepted tradeoff for a local, single-user tool, but it should never
be committed to version control (see the .gitignore entry added alongside
this module) or shared.

Each slot can point at a model from any of the three supported providers
(Anthropic, OpenAI, Google), and each provider needs its own kind of API key.
So keys are stored per-slot *per-provider* — `api_keys: {"anthropic": "...",
"openai": "..."}` — rather than one key per slot. That way switching a slot's
model from a Claude model to a GPT model doesn't silently keep offering the
old Anthropic key as if it were valid for OpenAI, and switching back later
doesn't lose the key you'd already entered for the other provider.

Nothing here talks to the network; it only reads/writes local JSON and
produces a ModelConfig for the rest of the pipeline to use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import ModelConfig, provider_for, default_api_key_env_for

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default location before any project has been activated; the webapp repoints
# this at the active project's own directory via set_base_dir() on startup
# and whenever the user switches projects, so each project keeps its own
# separate API keys/model choices rather than sharing one global file.
SETTINGS_PATH = PROJECT_ROOT / "ai_settings.json"

_SLOTS = ("cheap", "main", "validation")


def set_base_dir(base_dir) -> None:
    """Repoint SETTINGS_PATH at `<base_dir>/ai_settings.json` — called by the
    webapp when the active project changes."""
    global SETTINGS_PATH
    SETTINGS_PATH = Path(base_dir) / "ai_settings.json"


def _read_raw() -> Dict[str, Any]:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_model_config() -> ModelConfig:
    """Build a ModelConfig using saved settings where present, falling back to
    the defaults in config.py (which in turn fall back to env vars for the
    API key at call time)."""
    cfg = ModelConfig()
    data = _read_raw()
    for slot in _SLOTS:
        saved = data.get(slot) or {}
        stage = getattr(cfg, slot)
        if saved.get("model"):
            stage.model = saved["model"]
        # api_key_env always tracks whichever provider the current model
        # belongs to, so fallback-to-environment-variable messaging and
        # resolution stay correct even after switching providers.
        stage.api_key_env = default_api_key_env_for(stage.model)
        provider = provider_for(stage.model)
        api_keys = saved.get("api_keys") or {}
        if api_keys.get(provider):
            stage.api_key_override = api_keys[provider]
        if saved.get("effort"):
            stage.effort = saved["effort"]
    return cfg


def load_masked() -> Dict[str, Any]:
    """Settings for display in the UI — API keys are masked, never returned in
    full once saved. `effort` isn't secret, so it's returned as-is (None if
    never set, meaning "use the API default"). `has_key`/`key_preview` reflect
    the key saved for the *current* model's provider specifically."""
    cfg = ModelConfig()  # for default model names when nothing saved yet
    data = _read_raw()
    out: Dict[str, Any] = {}
    for slot in _SLOTS:
        saved = data.get(slot) or {}
        model = saved.get("model") or getattr(cfg, slot).model
        provider = provider_for(model)
        key = (saved.get("api_keys") or {}).get(provider) or ""
        out[slot] = {
            "model": model,
            "provider": provider,
            "api_key_env": default_api_key_env_for(model),
            "effort": saved.get("effort") or None,
            "has_key": bool(key),
            "key_preview": ("•" * 4 + key[-4:]) if len(key) >= 4 else ("•" * len(key) if key else ""),
        }
    return out


def save(payload: Dict[str, Any]) -> None:
    """Update settings. `payload` is
    {"cheap": {"model": ..., "api_key": ..., "effort": ...}, ...} for any
    subset of slots/fields.

    `api_key` is stored against whichever provider the *resulting* model
    (the incoming "model" if given, else whatever was already saved) belongs
    to. An empty/missing api_key leaves that provider's previously saved key
    untouched (so the UI can round-trip a masked field without accidentally
    wiping a real key); pass {"api_key": null} to explicitly clear it.
    Passing "effort": null or "" clears it (falls back to the API default).
    """
    existing = _read_raw()
    for slot in _SLOTS:
        if slot not in payload:
            continue
        existing.setdefault(slot, {})
        existing[slot].setdefault("api_keys", {})
        incoming = payload[slot] or {}

        if "model" in incoming and incoming["model"]:
            existing[slot]["model"] = incoming["model"]

        current_model = existing[slot].get("model", getattr(ModelConfig(), slot).model)
        provider = provider_for(current_model)

        if "api_key" in incoming:
            if incoming["api_key"] is None:
                existing[slot]["api_keys"][provider] = ""
            elif incoming["api_key"]:  # non-empty string overwrites
                existing[slot]["api_keys"][provider] = incoming["api_key"]
            # empty string / omitted -> leave that provider's key as-is

        if "effort" in incoming:
            existing[slot]["effort"] = incoming["effort"] or ""
    SETTINGS_PATH.write_text(json.dumps(existing, indent=2))
