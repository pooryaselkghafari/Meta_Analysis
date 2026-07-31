"""Meta-analysis extraction pipeline for macroeconomic papers.

Stages 1-5 (parsing, chunking, classification, filtering, extraction, validation).
Prompts are left blank in prompts.py; the LLM call structure is in place.
"""
from .config import (
    PipelineConfig, MarkerConfig, ChunkConfig, ModelConfig, AIModelConfig,
    AVAILABLE_MODELS, MODEL_EFFORT_LEVELS, effort_levels_for,
    PROVIDER_API_KEY_ENV, provider_for, default_api_key_env_for,
)
from .pipeline import Pipeline
from .llm import LLMClient
from . import settings_store
from . import prompts_store
from .models import (
    Chunk, ChunkType, ExtractionRecord, ParsedPaper, PaperMetadata,
    TransformationType, MatchType, SpecificationStatus,
    VariableRole, EstimateType, SourceType,
)

__all__ = [
    "Pipeline", "PipelineConfig", "MarkerConfig", "ChunkConfig", "ModelConfig",
    "AIModelConfig", "AVAILABLE_MODELS", "MODEL_EFFORT_LEVELS", "effort_levels_for",
    "PROVIDER_API_KEY_ENV", "provider_for", "default_api_key_env_for",
    "LLMClient", "settings_store", "prompts_store",
    "Chunk", "ChunkType", "ExtractionRecord", "ParsedPaper", "PaperMetadata",
    "TransformationType", "MatchType", "SpecificationStatus",
    "VariableRole", "EstimateType", "SourceType",
]
__version__ = "0.1.0"