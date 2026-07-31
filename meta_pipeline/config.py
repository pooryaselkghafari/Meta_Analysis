"""Central configuration for the pipeline.

Nothing here is secret — API keys are read from the environment, not stored.
Tune chunking, model names, and paths in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarkerConfig:
    """Stage 1 — Marker PDF parsing."""
    # Force OCR for scanned / older working papers (NBER, IMF). Off by default.
    force_ocr: bool = False
    # Marker output format; the pipeline expects markdown with tables preserved.
    output_format: str = "markdown"
    # Max concurrent PDF conversions.
    workers: int = 2
    # If Marker is unavailable, allow a plaintext fallback (pdfminer/pypdf).
    allow_fallback: bool = True


@dataclass
class ChunkConfig:
    """Stage 3 — table-aware chunking."""
    # Paragraphs of context kept on each side of a detected table.
    paras_before_table: int = 2
    paras_after_table: int = 2
    # For non-table prose, target chunk size (characters) with overlap.
    prose_target_chars: int = 1800
    prose_overlap_chars: int = 200
    # Regex/heuristic signals that a chunk may carry an estimate.
    estimate_keywords: List[str] = field(default_factory=lambda: [
        "coefficient", "elasticity", "multiplier", "estimate", "effect",
        "standard error", "std. err", "p-value", "p <", "95% ci",
        "confidence interval", "significant",
    ])


# Single source of truth for which model strings are valid — used to populate
# the Settings page's model dropdowns, so a typo'd model name (which would
# otherwise only surface as a confusing API error later) isn't possible from
# the UI. Three providers are supported: Anthropic (Claude), OpenAI (GPT), and
# Google (Gemini). Keep this in sync with the model strings each provider
# actually serves.
AVAILABLE_MODELS: List[Dict[str, str]] = [
    # --- Anthropic ---
    {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "provider": "anthropic"},
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "provider": "anthropic"},
    {"id": "claude-opus-4-8", "label": "Claude Opus 4.8", "provider": "anthropic"},
    {"id": "claude-fable-5", "label": "Claude Fable 5", "provider": "anthropic"},
    # --- OpenAI ---
    {"id": "gpt-5.6-sol", "label": "GPT-5.6 Sol", "provider": "openai"},
    {"id": "gpt-5.6-terra", "label": "GPT-5.6 Terra", "provider": "openai"},
    {"id": "gpt-5.6-luna", "label": "GPT-5.6 Luna", "provider": "openai"},
    # --- Google ---
    {"id": "gemini-3.1-pro-preview", "label": "Gemini 3.1 Pro (preview)", "provider": "google"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "provider": "google"},
    {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "provider": "google"},
    {"id": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash-Lite", "provider": "google"},
]

# provider -> default env var holding that provider's API key.
PROVIDER_API_KEY_ENV: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}

_MODEL_PROVIDERS: Dict[str, str] = {m["id"]: m["provider"] for m in AVAILABLE_MODELS}


def provider_for(model: str) -> str:
    """Which provider a model string belongs to. Defaults to "anthropic" for
    an unrecognized/hand-edited model string, matching the original
    single-provider behavior."""
    return _MODEL_PROVIDERS.get(model, "anthropic")


def default_api_key_env_for(model: str) -> str:
    return PROVIDER_API_KEY_ENV.get(provider_for(model), "ANTHROPIC_API_KEY")


# Which "effort" / "thinking" levels each model accepts, and under what
# parameter name — this differs per provider:
#   - Anthropic: output_config={"effort": <level>}
#   - OpenAI:    reasoning={"effort": <level>}          (Responses API)
#   - Google:    generation_config={"thinking_level": <level>}
# llm.py picks the right parameter shape based on provider_for(model); this
# map only says which *level strings* are valid for a given model, and is used
# both to build the Settings page's effort dropdown and to guard against
# sending an unsupported value (Haiku supports none of these levels at all).
MODEL_EFFORT_LEVELS: Dict[str, List[str]] = {
    # Anthropic
    "claude-sonnet-5": ["low", "medium", "high", "xhigh", "max"],
    "claude-opus-4-8": ["low", "medium", "high", "xhigh", "max"],
    "claude-fable-5": ["low", "medium", "high", "xhigh", "max"],
    # "claude-haiku-4-5-20251001" intentionally omitted — not supported.
    # OpenAI (GPT-5.6 family)
    "gpt-5.6-sol": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-5.6-terra": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-5.6-luna": ["none", "low", "medium", "high", "xhigh", "max"],
    # Google (Gemini "thinking_level")
    "gemini-3.1-pro-preview": ["low", "medium", "high"],
    "gemini-3.5-flash": ["minimal", "low", "medium", "high"],
    "gemini-2.5-flash": ["low", "medium", "high"],
    "gemini-2.5-flash-lite": ["low", "medium", "high"],
}


def effort_levels_for(model: str) -> List[str]:
    """Valid effort/thinking-level values for a given model, or [] if it
    doesn't support any such parameter at all."""
    return MODEL_EFFORT_LEVELS.get(model, [])


@dataclass
class AIModelConfig:
    """One independently configurable AI slot: its own model name, its own
    API key, and its own effort level. Three of these exist (see ModelConfig
    below) so each can be pointed at a different model — or even a different
    provider's key — without the others being affected.

    api_key resolution order: an explicit override (typically set via the
    Settings page and persisted by meta_pipeline.settings_store) wins; falling
    back to the environment variable named by api_key_env. Nothing is ever
    written to disk from this class itself — persistence is settings_store's
    job — so importing/using PipelineConfig never has a side effect.

    `effort` is None by default, meaning "omit the parameter" (the API then
    uses its own default, currently "high"). It's only ever sent on an actual
    call if it's both set and in effort_levels_for(self.model) — see
    LLMClient._call — so pointing a slot at a model that doesn't support
    effort (e.g. Haiku) just silently ignores a stale effort value rather than
    erroring.
    """
    label: str
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_key_override: Optional[str] = None
    effort: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        if self.api_key_override:
            return self.api_key_override
        # api_key_env is kept in sync with the current model's provider by
        # settings_store.load_model_config(), but fall back to computing it
        # fresh here too — this property should never point at the wrong
        # provider's env var just because api_key_env is stale.
        env_name = self.api_key_env or default_api_key_env_for(self.model)
        return os.environ.get(env_name) or os.environ.get(default_api_key_env_for(self.model))


@dataclass
class ModelConfig:
    """LLM endpoints. Three independently configurable slots, matching the
    three AI stages in the pipeline:
      - cheap:      Stage 2 classification + Stage 3b detection (same slot,
                    since both are cheap/binary-ish calls over many chunks)
      - main:       Stage 4 merged extraction (the expensive, high-quality call)
      - validation: Stage 5 validation pass (cross-checks Stage 4's output;
                    kept separate from `main` so it can run on a cheaper/faster
                    model, or simply use its own API key/rate limit)

    Prompts live in prompts.py (left blank for now).
    """
    cheap: AIModelConfig = field(default_factory=lambda: AIModelConfig(
        label="cheap", model="claude-haiku-4-5-20251001"))
    main: AIModelConfig = field(default_factory=lambda: AIModelConfig(
        label="main", model="claude-opus-4-8"))
    validation: AIModelConfig = field(default_factory=lambda: AIModelConfig(
        label="validation", model="claude-sonnet-5"))
    max_tokens: int = 2000
    temperature: float = 0.0

    def slots(self) -> Dict[str, AIModelConfig]:
        return {"cheap": self.cheap, "main": self.main, "validation": self.validation}


@dataclass
class PipelineConfig:
    input_dir: str = "input_papers"
    output_dir: str = "output"
    # User-supplied targets + ontology (loaded elsewhere; paths only here).
    targets_path: str = "targets.json"
    ontology_path: str = "ontology.json"

    marker: MarkerConfig = field(default_factory=MarkerConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Global switch: run LLM stages or stop after deterministic chunking (useful
    # while prompts are still blank).
    run_llm_stages: bool = False
