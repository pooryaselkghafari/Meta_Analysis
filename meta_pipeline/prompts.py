"""Prompt templates.

Each prompt is a function returning (system, user) strings so that context
(targets, ontology, chunk text) can be injected at call time. The actual
prompt text is no longer hardcoded here — it's stored in `prompts_settings.json`
at the project root and editable from the Settings page (see
meta_pipeline.prompts_store). These functions just declare *which* placeholders
each stage supports and delegate the lookup + fill to that store.

While a stage's template is still blank (the out-of-the-box state), this
returns ("", ""), and llm.py's blank-prompt check short-circuits the API call —
same behavior as before prompts became UI-editable.
"""
from __future__ import annotations

from typing import Optional, Tuple

from . import prompts_store


# --------------------------------------------------------------------------- #
# Stage 2 — cheap model: classify a chunk by content type
# --------------------------------------------------------------------------- #
def classification_prompt(chunk_text: str) -> Tuple[str, str]:
    """Classify chunk into one of ChunkType values.

    Should instruct the model to return a single label (regression_table /
    result_text / methodology / data_description / literature_review / other).
    Template placeholders: {{chunk_text}}.
    """
    return prompts_store.render("classification", chunk_text=chunk_text)


# --------------------------------------------------------------------------- #
# Stage 3 — cheap model: does this chunk contain a relevant point estimate?
# --------------------------------------------------------------------------- #
def _format_targets_for_prompt(targets: dict) -> str:
    """Plain-text rendering of the target elasticity types + products list,
    rather than handing the model a raw JSON blob. A cheap model parses
    "Elasticity type(s): X, Y" faster and more reliably than nested
    {"elasticities": [...], "products": [...]} syntax — there's no structure
    to isolate, just two labeled lines.

    Cross-price elasticities are inherently about a PAIR of products (the
    good whose demand is explained, and the other good whose price moved) —
    there's no separate "pairs" list to configure; both products for a
    cross-price row are just drawn from this same flat product list, and it's
    the model's job (per the extraction prompt) to identify which combination
    a given excerpt actually reports on."""
    targets = targets or {}
    elasticities = targets.get("elasticities") or []
    products = targets.get("products") or []
    lines = [
        f"Elasticity type(s) to extract: {', '.join(elasticities) if elasticities else '(none specified)'}",
        f"Product(s) of interest: {', '.join(products) if products else '(none specified)'}",
    ]
    return "\n".join(lines)


def _format_classification_for_prompt(classification: Optional[dict]) -> str:
    """Plain-text rendering of Stage 2's classification output, same reasoning
    as _format_targets_for_prompt — labeled lines instead of raw JSON so the
    cheap model isolates each field instantly instead of having to parse
    syntax nested inside the rest of the prompt."""
    classification = classification or {}
    if not classification.get("primary") and not classification.get("labels"):
        return "(this chunk has not been classified by Stage 2 yet)"
    labels = classification.get("labels") or []
    return "\n".join([
        f"Primary content type: {classification.get('primary') or '(none)'}",
        f"All content types that apply: {', '.join(labels) if labels else '(none)'}",
        f"Stage 2 confidence: {classification.get('confidence') if classification.get('confidence') is not None else '(unknown)'}",
        f"Stage 2 keep decision: {classification.get('keep') if classification.get('keep') is not None else '(unknown)'}",
    ])


def detection_prompt(chunk_text: str, targets: dict,
                      classification: Optional[dict] = None) -> Tuple[str, str]:
    """Binary screen: does this chunk plausibly contain a point estimate for one
    of the user's target elasticity types, calculated on one of their target
    products? Cheap model.

    `classification` is Stage 2's own output for this chunk (labels, primary,
    confidence, keep) — handed over as context so Stage 3 can lean on it as a
    prior (e.g. primary="regression_table" is a strong signal) rather than
    re-deriving content type from nothing. May be None/empty when Stage 2
    hasn't run for this chunk yet.

    `targets` and `classification` are pre-formatted into plain labeled text
    (not raw JSON) before filling the template, to keep the prompt easy for a
    cheap model to parse instantly rather than making it isolate structure
    nested inside the rest of the prompt.
    Template placeholders: {{chunk_text}}, {{targets}}, {{classification}}.
    """
    return prompts_store.render(
        "detection", chunk_text=chunk_text,
        targets=_format_targets_for_prompt(targets),
        classification=_format_classification_for_prompt(classification),
    )


# --------------------------------------------------------------------------- #
# Stage 4a — main model: paper-level metadata, once per paper
# --------------------------------------------------------------------------- #
def paper_metadata_prompt(abstract: Optional[str], methodology_text: str) -> Tuple[str, str]:
    """Extract the handful of facts that are constant across nearly all of a
    paper's estimates — country/region, sample period, frequency, N, data
    source, primary model/method — ONCE per paper, from the abstract plus
    every chunk Stage 2 classified as methodology or data_description
    (concatenated by the caller into `methodology_text`, regardless of
    whether those chunks passed Stage 3's estimate-signal filter, since a
    paragraph stating a sample size rarely contains a coefficient itself).

    This exists so Stage 4's per-estimate call isn't asked to restate the
    same constants for every single coefficient it emits — wasted tokens,
    and a source of the model giving a slightly different N or year range
    each time it's asked.
    Template placeholders: {{abstract}}, {{methodology_text}}.
    """
    return prompts_store.render(
        "paper_metadata",
        abstract=abstract or "(no abstract detected for this paper)",
        methodology_text=methodology_text or "(no methodology/data-description chunks available for this paper)",
    )


# --------------------------------------------------------------------------- #
# Stage 4 — main model: merged extraction into the strict schema
# --------------------------------------------------------------------------- #
def _format_ontology_for_prompt(ontology: Optional[dict]) -> str:
    """Plain-text rendering of any known variable-synonym mappings, same
    "token bleed" reasoning as _format_targets_for_prompt. `ontology` is
    optional and, in most projects so far, empty (ontology.json defaults to
    {}) — render an explicit "none configured" line rather than an empty
    JSON blob so the model doesn't waste effort trying to parse structure
    that isn't there."""
    if not ontology:
        return "(no variable ontology configured for this project — match target names against the paper's own terminology using judgment.)"
    lines = []
    for target, synonyms in ontology.items():
        if isinstance(synonyms, list):
            lines.append(f"- {target}: also accept {', '.join(str(s) for s in synonyms)}")
        else:
            lines.append(f"- {target}: {synonyms}")
    return "\n".join(lines) if lines else "(no variable ontology configured for this project.)"


def _format_screening_for_prompt(screening: Optional[dict]) -> str:
    """Plain-text rendering of Stages 2+3's combined output for this chunk —
    content classification (Stage 2) and target-relevance screening
    (Stage 3) — handed to the main model as prior context, same reasoning as
    _format_classification_for_prompt in the detection prompt."""
    screening = screening or {}
    if not screening.get("primary") and screening.get("detected") is None:
        return "(no prior classification/detection recorded for this chunk)"
    lines = [
        f"Stage 2 content type: {screening.get('primary') or '(none)'} "
        f"(confidence: {screening.get('classification_confidence') or 'unknown'})",
        f"Stage 3 target-relevant estimate detected: {screening.get('detected') if screening.get('detected') is not None else 'unknown'} "
        f"(confidence: {screening.get('detection_confidence') or 'unknown'})",
        f"Stage 3 keep reason: {screening.get('keep_reason') or '(none)'}",
    ]
    tm = screening.get("target_match") or {}
    elasticity_match = tm.get("elasticity_match") or {}
    product_match = tm.get("product_match") or {}
    cross_price_match = tm.get("cross_price_product_match") or {}
    if elasticity_match:
        lines.append(f"Stage 3 elasticity-type match: {elasticity_match.get('match_type') or '?'} — paper term: {elasticity_match.get('paper_term') or '(none)'}")
    if product_match:
        lines.append(f"Stage 3 product match: {product_match.get('match_type') or '?'} — paper term: {product_match.get('paper_term') or '(none)'}")
    if cross_price_match:
        lines.append(f"Stage 3 cross-price product match: {cross_price_match.get('match_type') or '?'} — paper term: {cross_price_match.get('paper_term') or '(none)'}")
    return "\n".join(lines)


def extraction_prompt(chunk_text: str, targets: dict, ontology: dict,
                       abstract: Optional[str] = None,
                       screening: Optional[dict] = None) -> Tuple[str, str]:
    """The main extraction call. Must produce JSON matching a LIST of
    ExtractionRecord-shaped estimates (a single excerpt, e.g. one
    regression table, can legitimately yield zero, one, or several distinct
    point estimates — see <tables_and_multiple_estimates> in the system
    prompt): variable matching (against targets/ontology), transformation
    flags (descriptive only, never converting), units, point estimate,
    specification status with evidence, and self-flagged review reasons.

    Two categories of field are deliberately NOT asked of this per-estimate
    call:
    - Provenance beyond row/column (page, table id, table_complete,
      pages_used) — already known deterministically from the chunk, filled
      in by Pipeline._record_from_raw.
    - Paper-level facts (model_type, countries_region, time_period,
      frequency, n_obs, n_units, data_source) — constant across nearly all
      of a paper's estimates, so asking for them on every single coefficient
      wastes tokens and invites inconsistency. These are established ONCE
      per paper by paper_metadata_prompt/Stage 4a and backfilled onto each
      record by Pipeline._backfill_paper_metadata. The per-estimate prompt
      only asks for them when THIS SPECIFIC excerpt states something that
      diverges from the paper's norm (e.g. a robustness check run on a
      different subsample) — otherwise leave them null.

    `abstract` is the paper's abstract (ParsedPaper.abstract) — whole-paper
    context only, never itself a source of estimates. `screening` bundles
    Stage 2's classification and Stage 3's detection output for this same
    chunk (see _format_screening_for_prompt) — a prior to lean on, not a
    verdict to defer to, since both earlier stages are deliberately
    recall-biased and expected to pass along some chunks that, on a careful
    read, don't actually contain a usable target estimate.

    `targets`, `ontology`, and `screening` are pre-formatted into plain
    labeled text (not raw JSON) before filling the template, same "token
    bleed" reasoning as the detection prompt.
    Template placeholders: {{chunk_text}}, {{targets}}, {{ontology}},
    {{abstract}}, {{screening}}.
    """
    return prompts_store.render(
        "extraction", chunk_text=chunk_text,
        targets=_format_targets_for_prompt(targets),
        ontology=_format_ontology_for_prompt(ontology),
        abstract=abstract or "(no abstract detected for this paper)",
        screening=_format_screening_for_prompt(screening),
    )


# --------------------------------------------------------------------------- #
# Stage 5 — validation pass
# --------------------------------------------------------------------------- #
def validation_prompt(record_json: str, chunk_text: str) -> Tuple[str, str]:
    """Cross-check an extracted record against its source chunk: catch misreads,
    wrong-column errors, and transformation-flag contradictions. Should return the
    record with corrected fields and/or requires_review + review_reason set.
    Template placeholders: {{record_json}}, {{chunk_text}}.
    """
    return prompts_store.render("validation", record_json=record_json, chunk_text=chunk_text)
