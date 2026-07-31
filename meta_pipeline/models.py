"""Data models for the meta-analysis extraction pipeline.

These mirror the consolidated schema from the design document (v4). They are the
contract between stages: the LLM stages emit these, the statistical layer consumes
them. Nothing here performs transformations — the transformation fields are purely
descriptive metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


# --------------------------------------------------------------------------- #
# Enums / controlled vocabularies
# --------------------------------------------------------------------------- #
class ChunkType(str, Enum):
    """Stage 2 content classification. A+B+C+D are kept for extraction; E is dropped."""
    REGRESSION_TABLE = "regression_table"        # A
    RESULT_TEXT = "result_text"                  # B
    METHODOLOGY = "methodology"                  # C
    DATA_DESCRIPTION = "data_description"        # D
    LITERATURE_REVIEW = "literature_review"      # E (excluded)
    OTHER = "other"


class TransformationType(str, Enum):
    """Descriptive only. The pipeline never converts between these forms."""
    NONE = "none"
    LOG = "log"
    LOG_DIFFERENCE = "log_difference"
    RATIO = "ratio"
    PERCENT_CHANGE = "percent_change"
    FIRST_DIFFERENCE = "first_difference"
    GROWTH_RATE = "growth_rate"
    INDEX = "index"
    STANDARDIZED = "standardized"
    OTHER = "other"
    UNKNOWN = "unknown"


class MatchType(str, Enum):
    EXACT = "exact"
    SYNONYM = "synonym"
    RELATED_SEPARATE = "related_separate"
    NO_MATCH = "no_match"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SpecificationStatus(str, Enum):
    EXPLICIT_BASELINE = "explicit_baseline"
    INFERRED_BASELINE = "inferred_baseline"
    UNKNOWN = "unknown"


class UnitSource(str, Enum):
    TABLE = "table"
    DATA_SECTION = "data_section"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class VariableRole(str, Enum):
    """Replaces the earlier boolean is_main_independent_variable, which asked
    the model to judge something often genuinely ambiguous (which of several
    plausible regressors is "the" main one). This is the observable version:
    treatment/primary_regressor are both "the paper cares about this
    variable's effect", just standard terminology for experimental/quasi-
    experimental vs. observational designs; control is an ancillary regressor
    that happens to share wording with a target; unknown when the excerpt
    doesn't give enough to tell."""
    TREATMENT = "treatment"
    PRIMARY_REGRESSOR = "primary_regressor"
    CONTROL = "control"
    UNKNOWN = "unknown"


class EstimateType(str, Enum):
    """What kind of quantity `coefficient` actually is — these are not
    interchangeable objects for a meta-analysis (an elasticity and a raw
    regression coefficient can't be pooled together without knowing which is
    which)."""
    REGRESSION_COEFFICIENT = "regression_coefficient"
    ELASTICITY = "elasticity"
    SEMI_ELASTICITY = "semi_elasticity"
    MULTIPLIER = "multiplier"
    IMPULSE_RESPONSE = "impulse_response"
    MARGINAL_EFFECT = "marginal_effect"
    HAZARD_RATIO = "hazard_ratio"
    OTHER = "other"


class SourceType(str, Enum):
    TABLE = "table"
    TEXT = "text"
    FIGURE = "figure"


# --------------------------------------------------------------------------- #
# Structural / intermediate models (produced by the deterministic stages)
# --------------------------------------------------------------------------- #
@dataclass
class SourceLocation:
    page: Optional[int] = None
    table: Optional[str] = None
    column: Optional[str] = None
    row: Optional[str] = None
    text_anchor: Optional[str] = None


@dataclass
class Chunk:
    """A unit of text passed between stages. Produced by Stage 1/3 chunking,
    classified in Stage 2, filtered in Stage 3, and fed to the LLM in Stage 4."""
    chunk_id: str
    paper_id: str
    text: str
    # populated by parsing (Stage 1)
    contains_table: bool = False
    table_complete: Optional[bool] = None
    pages_used: List[int] = field(default_factory=list)
    source_location: SourceLocation = field(default_factory=SourceLocation)
    # True if this chunk is (or contains) the paper's abstract. Set during
    # chunking (Stage 3a) by matching against ParsedPaper.abstract. Abstract
    # chunks are exempted from the heuristic estimate-signal filter — see
    # Pipeline.filter_chunks — since an abstract almost never contains a point
    # estimate but is still valuable whole-paper context and shouldn't be shown
    # as "dropped" in audit views.
    is_abstract: bool = False
    # populated by classification (Stage 2). A chunk can genuinely belong to
    # more than one category (e.g. a paragraph that both names its estimation
    # strategy and reports the resulting coefficient) — forcing a single label
    # would silently discard whichever secondary signal didn't win. `labels`
    # holds every category the classifier judged applicable; `chunk_type`
    # holds just the primary/dominant one, kept as a single field for
    # backward compatibility with anything that only cares about one label
    # (e.g. simple UI display).
    chunk_type: Optional[ChunkType] = None
    labels: List[ChunkType] = field(default_factory=list)
    # "high"/"medium"/"low" — a coarse enum rather than a float, since cheap
    # models self-calibrate categories far more reliably than a 0.0-1.0 score.
    classification_confidence: Optional[str] = None
    # The classifier's own keep/discard call (True unless the chunk is purely
    # "other", or purely literature review with no signal about this paper's
    # own results) — Stage 3 filtering treats this as authoritative when
    # present, rather than re-deriving a keep/discard decision from chunk_type
    # alone.
    keep_classification: Optional[bool] = None
    # populated by detection (Stage 3, cheap-LLM target screen). `target_match`
    # is the {"matches", "dv_match": {"paper_term","match_type"}, "iv_match": {...}}
    # structure from LLMClient._parse_detection_result — kept as a plain dict
    # (not a dataclass) since it's optional/nullable at every level and only
    # ever round-tripped through JSON, never computed on. `detected` is
    # Stage 3's own keep/discard call for this chunk (separate from
    # `keep_classification`, which is Stage 2's).
    detected: Optional[bool] = None
    detection_confidence: Optional[str] = None
    target_match: Optional[Dict[str, Any]] = None
    keep_reason: Optional[str] = None
    # populated by filtering (Stage 3)
    passes_filter: Optional[bool] = None
    filter_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.chunk_type is not None:
            d["chunk_type"] = self.chunk_type.value
        if self.labels:
            d["labels"] = [lbl.value if isinstance(lbl, Enum) else lbl for lbl in self.labels]
        return d


@dataclass
class ParsedPaper:
    """Output of Stage 1 for a single PDF."""
    paper_id: str
    source_path: str
    markdown: str
    chunks: List[Chunk] = field(default_factory=list)
    parse_ok: bool = True
    parse_error: Optional[str] = None
    # Best-effort abstract, extracted once per paper. Kept separately from the
    # chunking/filtering flow: the abstract rarely contains a point estimate, so
    # the heuristic filter would normally drop it, but it's the cheapest way to
    # give the main extraction LLM (Stage 4) whole-paper context (what the paper
    # is actually about) alongside each filtered chunk it sees in isolation.
    abstract: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction record (the main deliverable — Stage 4 emits, Stage 5 enriches)
# --------------------------------------------------------------------------- #
@dataclass
class TimePeriod:
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class PaperMetadata:
    """Paper-level facts that are constant across nearly all of a paper's
    estimates — country/region, sample period, frequency, N, data source, and
    primary model/method. Extracted ONCE per paper (Stage 4a, from the
    abstract plus every methodology/data_description chunk) rather than
    re-asked of the model for every single coefficient it emits in Stage 4 —
    asking a per-estimate call to restate the same constants ten or twenty
    times over is wasted tokens and an avoidable source of inconsistency
    (the model giving a slightly different N or year range each time).
    Pipeline._backfill_paper_metadata fills these into each ExtractionRecord
    only where that record's own chunk didn't already state something more
    specific (e.g. a robustness check run on a different subsample)."""
    paper_id: str
    model_type: Optional[str] = None
    countries_region: Optional[str] = None
    time_period: TimePeriod = field(default_factory=TimePeriod)
    frequency: Optional[str] = None
    n_obs: Optional[int] = None
    n_units: Optional[int] = None
    data_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionRecord:
    """One record per extracted estimate. Matches section 6 of the design doc."""
    paper_id: str
    estimate_id: str
    # The chunk this record was extracted from. A single chunk (e.g. a full
    # regression table) can yield several records — this is how Stage 5
    # validation finds the right source chunk to cross-check against, since
    # estimate_id alone no longer maps 1:1 back to a chunk_id.
    source_chunk_id: Optional[str] = None

    # ---- variable matching ----
    # The pipeline's domain is elasticities of demand: each record matches one
    # of the user's target elasticity types (e.g. "Own-price Elasticity")
    # against one of their target products (e.g. "maize"). Cross-price
    # elasticities are inherently about TWO products — the good whose demand
    # is being explained, and the other good whose price moved — so
    # target_cross_price_product/paper_cross_price_product_wording_raw are
    # only ever populated on cross-price rows; both target_product and
    # target_cross_price_product must be exact entries from the user's own
    # product list, not free text, same as target_elasticity_type.
    target_elasticity_type: Optional[str] = None
    target_product: Optional[str] = None
    target_cross_price_product: Optional[str] = None
    paper_elasticity_wording_raw: Optional[str] = None
    paper_product_wording_raw: Optional[str] = None
    paper_cross_price_product_wording_raw: Optional[str] = None
    match_type: Optional[MatchType] = None
    target_match_justification: Optional[str] = None
    # Replaces the earlier boolean is_main_independent_variable — see
    # VariableRole's docstring for why an enum is more observable than a
    # "is this THE main variable" judgment call.
    variable_role: Optional[VariableRole] = None

    # ---- transformation flags (NO conversion performed) ----
    # A single pair now, not a DV/IV pair — the product is just a categorical
    # label (no raw/transformed state of its own); only the elasticity value
    # itself can be a direct estimate vs. one derived/converted from something
    # else (e.g. a semi-elasticity converted, or a log-log slope reported
    # as-is).
    elasticity_is_raw: Optional[bool] = None
    elasticity_transformation_type: Optional[TransformationType] = None

    # ---- units ----
    elasticity_unit: Optional[str] = None
    unit_source: Optional[UnitSource] = None

    # ---- point estimate ----
    estimate_type: Optional[EstimateType] = None
    coefficient: Optional[float] = None
    standard_error: Optional[float] = None
    standard_error_reported: Optional[bool] = None
    confidence_interval: Optional[List[float]] = None
    confidence_interval_reported: Optional[bool] = None
    p_value: Optional[float] = None
    p_value_reported: Optional[bool] = None
    significance_stars: Optional[str] = None
    # Some papers report only a t/z/chi2/F statistic rather than an SE or
    # p-value directly — {"type": "t"|"z"|"chi2"|"f"|"other", "value": float}
    # so that number isn't simply lost.
    test_statistic: Optional[Dict[str, Any]] = None

    # ---- specification ----
    specification_status: Optional[SpecificationStatus] = None
    baseline_evidence: Optional[str] = None
    # ---- paper-level fields (Stage 4a default, chunk-level override) ----
    model_type: Optional[str] = None
    countries_region: Optional[str] = None
    time_period: TimePeriod = field(default_factory=TimePeriod)
    frequency: Optional[str] = None
    n_obs: Optional[int] = None
    n_units: Optional[int] = None
    data_source: Optional[str] = None

    # ---- provenance ----
    source_location: SourceLocation = field(default_factory=SourceLocation)
    source_type: Optional[SourceType] = None
    table_complete: Optional[bool] = None
    pages_used: List[int] = field(default_factory=list)

    # ---- review routing ----
    requires_review: bool = False
    review_reason: List[str] = field(default_factory=list)
    # Set by the webapp's manual-edit endpoint (Stage 4's "directly edit the
    # table" feature) when a human corrects any field on this record. Never
    # set by the extraction stage itself.
    manually_edited: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # unwrap enums to their string values for clean JSON
        for k, v in list(d.items()):
            if isinstance(getattr(self, k, None), Enum):
                d[k] = getattr(self, k).value
        return d

    # ----- consistency validation (Stage 5 support) -----
    def check_transformation_consistency(self) -> List[str]:
        """Enforce the design-doc rule: is_raw==True must pair with type 'none',
        and any other type implies is_raw==False. Returns list of problems found."""
        problems: List[str] = []
        is_raw, ttype = self.elasticity_is_raw, self.elasticity_transformation_type
        if is_raw is not None and ttype is not None:
            if is_raw and ttype != TransformationType.NONE:
                problems.append(
                    f"elasticity_is_raw is True but elasticity_transformation_type is "
                    f"'{ttype.value}' (expected 'none')"
                )
            if not is_raw and ttype == TransformationType.NONE:
                problems.append(
                    "elasticity_is_raw is False but elasticity_transformation_type is 'none'"
                )
        return problems
