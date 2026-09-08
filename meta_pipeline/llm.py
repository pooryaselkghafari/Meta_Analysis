"""LLM client — two tiers (cheap detection/classification, main extraction),
across three possible providers (Anthropic, OpenAI, Google).

All prompt text comes from prompts.py (blank for now), so these functions are
the *structure* of the calls: they assemble a system+user prompt, invoke the
right model on whichever provider it belongs to, and parse JSON out of the
response. When prompts are empty the calls short-circuit and return None, so
the pipeline can run end-to-end during development without hitting the API.

Each AIModelConfig slot can point at a model from any of the three providers;
provider_for(stage.model) decides which SDK/request-shape to use. Provider
packages (openai, google-genai) are only imported lazily, on first use of a
slot pointed at that provider — so a Claude-only setup never needs them
installed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .config import AIModelConfig, ModelConfig, effort_levels_for, provider_for
from . import prompts


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # One provider client per AI slot (cheap/main/validation), lazily
        # created — each slot can carry its own API key and even its own
        # provider, so they can't safely share a single client instance.
        self._clients: Dict[str, Any] = {}

    # ----- low level: client construction, one branch per provider -----
    def _ensure_client(self, stage: AIModelConfig):
        if stage.label in self._clients:
            return self._clients[stage.label]
        provider = provider_for(stage.model)
        if not stage.api_key:
            raise RuntimeError(
                f"No API key set for the '{stage.label}' AI (model {stage.model}, "
                f"provider {provider}) — set it on the Settings page, or via env "
                f"var {stage.api_key_env}"
            )
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError("anthropic package not installed (pip install anthropic)") from e
            client = anthropic.Anthropic(api_key=stage.api_key)
        elif provider == "openai":
            try:
                import openai
            except ImportError as e:
                raise RuntimeError("openai package not installed (pip install openai)") from e
            client = openai.OpenAI(api_key=stage.api_key)
        elif provider == "google":
            try:
                from google import genai
            except ImportError as e:
                raise RuntimeError("google-genai package not installed (pip install google-genai)") from e
            client = genai.Client(api_key=stage.api_key)
        else:
            raise RuntimeError(f"Unknown provider '{provider}' for model {stage.model}")
        self._clients[stage.label] = client
        return client

    # ----- low level: request/response, one branch per provider -----
    def _call(self, stage: AIModelConfig, system: str, user: str,
              max_tokens: Optional[int] = None) -> str:
        client = self._ensure_client(stage)
        provider = provider_for(stage.model)
        tokens = max_tokens or self.cfg.max_tokens
        # An effort/thinking-level value only gets sent if it's both set AND
        # in effort_levels_for(stage.model) for the current model — sending it
        # to an unsupported model (e.g. Haiku) would be an API error, so a
        # stale value left over from switching a slot to an unsupported model
        # is silently dropped rather than breaking calls.
        effort = stage.effort if stage.effort in effort_levels_for(stage.model) else None
        # Newer effort/thinking-capable models (the ones with a non-empty
        # effort_levels_for entry — Sonnet 5, Opus 4.8, Fable 5, the GPT-5.6
        # family, Gemini 3.x) have replaced temperature with the effort/
        # thinking-level control and reject the parameter outright ("`
        # temperature` is deprecated for this model"). Only send it to
        # older-style models that don't support effort at all (e.g. Haiku),
        # where it's still the only sampling knob available.
        send_temperature = not effort_levels_for(stage.model)

        if provider == "anthropic":
            return self._call_anthropic(client, stage, system, user, tokens, effort, send_temperature)
        elif provider == "openai":
            return self._call_openai(client, stage, system, user, tokens, effort)
        elif provider == "google":
            return self._call_google(client, stage, system, user, tokens, effort, send_temperature)
        raise RuntimeError(f"Unknown provider '{provider}' for model {stage.model}")

    @staticmethod
    def _call_with_kwarg_fallback(fn, kwargs: Dict[str, Any], max_attempts: int = 6):
        """Calls fn(**kwargs), and if the installed SDK's version of fn
        doesn't accept one of these keywords (a plain Python TypeError,
        distinct from the API itself rejecting a value), drops that keyword
        and retries. Provider SDKs (anthropic/openai/google-genai) change
        their accepted parameters across versions more often than this
        codebase gets updated to match, and this runs on whatever version
        happens to be installed on each user's machine — rather than crash
        the whole call over one now-unsupported sampling knob like
        `temperature` or `thinking_config`, degrade gracefully and still get
        a response. `max_attempts` just bounds the loop; in practice at most
        a couple of keywords are ever dropped."""
        kwargs = dict(kwargs)
        for _ in range(max_attempts):
            try:
                return fn(**kwargs)
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                if not m or m.group(1) not in kwargs:
                    raise
                kwargs.pop(m.group(1))
        return fn(**kwargs)

    def _call_anthropic(self, client, stage, system, user, tokens, effort, send_temperature) -> str:
        kwargs: Dict[str, Any] = dict(
            model=stage.model,
            max_tokens=tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if send_temperature:
            kwargs["temperature"] = self.cfg.temperature
        if effort:
            kwargs["output_config"] = {"effort": effort}
        resp = self._call_with_kwarg_fallback(client.messages.create, kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

    def _call_openai(self, client, stage, system, user, tokens, effort) -> str:
        # Responses API: instructions carries the system prompt, input the
        # user prompt. `reasoning={"effort": ...}` mirrors Anthropic's effort
        # parameter for the GPT-5.6 family. temperature was never sent here
        # to begin with (the Responses API doesn't take it alongside
        # reasoning effort), so no change needed for the same underlying issue.
        kwargs: Dict[str, Any] = dict(
            model=stage.model,
            instructions=system,
            input=user,
            max_output_tokens=tokens,
        )
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        resp = self._call_with_kwarg_fallback(client.responses.create, kwargs)
        return resp.output_text

    def _call_google(self, client, stage, system, user, tokens, effort, send_temperature) -> str:
        # Gemini's generate_content: system prompt goes in
        # GenerateContentConfig.system_instruction, effort in
        # thinking_config.thinking_level (Gemini calls it a "thinking level",
        # not "effort", but it's the same trade-off — and, per the same
        # reasoning as Anthropic above, thinking-capable Gemini models may
        # likewise reject temperature alongside a thinking level).
        from google.genai import types

        config_kwargs: Dict[str, Any] = dict(
            system_instruction=system,
            max_output_tokens=tokens,
        )
        if send_temperature:
            config_kwargs["temperature"] = self.cfg.temperature
        if effort:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=effort)
        config = self._call_with_kwarg_fallback(types.GenerateContentConfig, config_kwargs)
        resp = client.models.generate_content(
            model=stage.model,
            contents=user,
            config=config,
        )
        return resp.text or ""

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from a model response, tolerating code fences."""
        if not text:
            return None
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # try to grab the outermost {...}
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _prompt_blank(system: str, user: str) -> bool:
        return not system.strip() and not user.strip()

    # Valid Stage 2 labels (mirrors ChunkType). Kept as plain strings here
    # rather than importing the enum, so llm.py stays decoupled from models.py.
    _CLASSIFICATION_LABELS = (
        "regression_table", "result_text", "methodology",
        "data_description", "literature_review", "other",
    )
    # Shared "high"/"medium"/"low" confidence enum, used by both Stage 2
    # (classification) and Stage 3 (detection) — cheap models self-calibrate
    # this far more reliably than a 0.0-1.0 float, which they tend to either
    # hallucinate at the extremes or use as a vague escape hatch.
    _CONFIDENCE_LEVELS = ("high", "medium", "low")

    @classmethod
    def _clean_label(cls, raw: str) -> Optional[str]:
        label = raw.strip().strip(".").strip("`").lower()
        return label if label in cls._CLASSIFICATION_LABELS else None

    @classmethod
    def _parse_classification_result(cls, raw: str) -> Optional[Dict[str, Any]]:
        """Defensive parsing for Stage 2's JSON-structured output.

        Current schema asks for:
            {"labels": [...], "primary": "...", "confidence": "high"/"medium"/"low", "keep": true/false}

        Confidence is a coarse enum rather than a float, same reasoning as
        Stage 3 (detection): cheap models self-calibrate "high/medium/low"
        far more reliably than a 0.0-1.0 number, which they tend to either
        hallucinate at the extremes or use as a vague escape hatch. A legacy
        numeric confidence (from before this was an enum) is tolerated but
        discarded rather than coerced, since a stray old float isn't
        trustworthy either way.

        Cheap models occasionally ignore "JSON only, no markdown" instructions
        (code fences, trailing punctuation, conversational preamble), and some
        older/weaker models only manage the legacy single-label shape
        {"label": "..."}. Handle all of that, and always return a dict with
        all four keys populated (falling back to sane derived defaults for
        anything the model didn't provide) rather than making callers handle
        several different possible shapes.
        """
        if not raw:
            return None

        parsed = cls._parse_json(raw)
        labels: list = []
        primary: Optional[str] = None
        confidence: Optional[str] = None
        keep: Optional[bool] = None

        if isinstance(parsed, dict):
            # labels: accept a list, or fall back to a single "label" string
            raw_labels = parsed.get("labels")
            if isinstance(raw_labels, list):
                for item in raw_labels:
                    if isinstance(item, str):
                        cleaned = cls._clean_label(item)
                        if cleaned and cleaned not in labels:
                            labels.append(cleaned)
            if isinstance(parsed.get("label"), str):
                cleaned = cls._clean_label(parsed["label"])
                if cleaned and cleaned not in labels:
                    labels.append(cleaned)

            # primary: explicit "primary" field, else first label
            if isinstance(parsed.get("primary"), str):
                primary = cls._clean_label(parsed["primary"])
            if primary is None and labels:
                primary = labels[0]

            # confidence: "high"/"medium"/"low" only — a legacy numeric value
            # (or anything else) is simply discarded, not coerced into a level.
            raw_conf = parsed.get("confidence")
            if isinstance(raw_conf, str) and raw_conf.strip().lower() in cls._CONFIDENCE_LEVELS:
                confidence = raw_conf.strip().lower()

            # keep: explicit boolean if present
            raw_keep = parsed.get("keep")
            if isinstance(raw_keep, bool):
                keep = raw_keep

        # Fallback: strict JSON parsing failed entirely, or produced no usable
        # labels — fall back to a plain substring search for a valid label
        # anywhere in the raw text, rather than dropping the chunk over a
        # formatting slip.
        if not labels:
            lowered = raw.lower()
            for label in cls._CLASSIFICATION_LABELS:
                if label in lowered:
                    labels.append(label)
            if labels:
                primary = labels[0]

        if not labels:
            return None

        # Derive a keep default when the model didn't supply one: exclude
        # only if "other" is the sole label (nothing worth extracting) or
        # "literature_review" is the sole label (no signal about this paper's
        # own results) — mirrors the pre-existing filter_chunks heuristic.
        if keep is None:
            keep = not (len(labels) == 1 and labels[0] in ("other", "literature_review"))

        return {
            "labels": labels,
            "primary": primary,
            "confidence": confidence,
            "keep": keep,
        }

    # ----- Stage 2: classification (cheap) -----
    def classify_chunk(self, chunk_text: str) -> Optional[Dict[str, Any]]:
        """Returns {"labels": [...], "primary": "...", "confidence": float|None,
        "keep": bool} or None if the prompt is blank or nothing parseable came
        back."""
        system, user = prompts.classification_prompt(chunk_text)
        if self._prompt_blank(system, user):
            return None
        out = self._call(self.cfg.cheap, system, user, max_tokens=200)
        return self._parse_classification_result(out)

    # Coarse variable-match hierarchy — "exact"/"semantic"/"conceptual" all
    # count as a real match; "related_but_not_equivalent" is deliberately its
    # own bucket rather than folded into no_match, since it's the case most
    # likely to cause a false positive (e.g. target "inflation" vs. paper
    # term "inflation volatility" — topically adjacent, not the same thing).
    _MATCH_TYPES = ("exact", "semantic", "conceptual", "related_but_not_equivalent", "no_match")
    # Why a chunk was kept/discarded — surfaced for debugging false
    # positives/negatives, not just used internally.
    _KEEP_REASONS = (
        "contains_target_estimate", "contains_target_variable_definition",
        "contains_model_specification_for_target_estimate",
        "literature_relevant_context", "literature_estimate_only",
        "off_target", "no_estimate_signal",
    )

    @classmethod
    def _parse_variable_match(cls, raw_match: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_match, dict):
            return None
        match_type = raw_match.get("match_type")
        if isinstance(match_type, str):
            match_type = match_type.strip().lower()
        if match_type not in cls._MATCH_TYPES:
            match_type = None
        paper_term = raw_match.get("paper_term")
        if not isinstance(paper_term, str):
            paper_term = None
        if match_type is None and paper_term is None:
            return None
        return {"paper_term": paper_term, "match_type": match_type}

    @classmethod
    def _parse_detection_result(cls, raw: str) -> Optional[Dict[str, Any]]:
        """Defensive parsing for Stage 3's JSON-structured output:
        {"detected": bool, "confidence": "high"/"medium"/"low",
        "target_match": {"matches": bool, "elasticity_match": {...},
        "product_match": {...}, "cross_price_product_match": {...}},
        "keep_reason": "..."}.

        Confidence is a coarse enum rather than a float — cheap models
        self-calibrate "high/medium/low" far more reliably than a 0.0-1.0
        number. target_match/keep_reason are optional/nullable throughout:
        a chunk with no product specified has product_match: null,
        cross_price_product_match is only ever populated for a cross-price
        elasticity chunk, a model that omits keep_reason entirely still gets
        a usable {"detected", "confidence"} result rather than nothing at
        all. Falls back to a plain yes/no substring search (the pre-JSON
        behavior) if strict parsing fails or the model only returns a bare
        word."""
        if not raw:
            return None
        parsed = cls._parse_json(raw)
        detected = None
        confidence = None
        target_match = None
        keep_reason = None
        if isinstance(parsed, dict) and isinstance(parsed.get("detected"), bool):
            detected = parsed["detected"]
            raw_conf = parsed.get("confidence")
            if isinstance(raw_conf, str) and raw_conf.strip().lower() in cls._CONFIDENCE_LEVELS:
                confidence = raw_conf.strip().lower()

            raw_tm = parsed.get("target_match")
            if isinstance(raw_tm, dict):
                elasticity_match = cls._parse_variable_match(raw_tm.get("elasticity_match"))
                product_match = cls._parse_variable_match(raw_tm.get("product_match"))
                cross_price_product_match = cls._parse_variable_match(raw_tm.get("cross_price_product_match"))
                matches = raw_tm.get("matches") if isinstance(raw_tm.get("matches"), bool) else None
                if elasticity_match or product_match or cross_price_product_match or matches is not None:
                    target_match = {
                        "matches": matches,
                        "elasticity_match": elasticity_match,
                        "product_match": product_match,
                        "cross_price_product_match": cross_price_product_match,
                    }

            raw_reason = parsed.get("keep_reason")
            if isinstance(raw_reason, str) and raw_reason.strip().lower() in cls._KEEP_REASONS:
                keep_reason = raw_reason.strip().lower()
        if detected is None:
            lowered = raw.strip().lower()
            if "true" in lowered or lowered.startswith("yes") or " yes" in lowered:
                detected = True
            elif "false" in lowered or lowered.startswith("no") or " no" in lowered:
                detected = False
            else:
                return None
        return {
            "detected": detected,
            "confidence": confidence,
            "target_match": target_match,
            "keep_reason": keep_reason,
        }

    # ----- Stage 3: detection (cheap) -----
    def detect_estimate(self, chunk_text: str, targets: dict,
                         classification: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Returns {"detected": bool, "confidence": "high"/"medium"/"low"/None,
        "target_match": {...}/None, "keep_reason": str/None}, or None if the
        prompt is blank or nothing parseable came back.

        `classification` is Stage 2's own output for this same chunk (labels,
        primary, confidence, keep) — passed through as extra context so this
        cheap screen doesn't have to re-derive from scratch what Stage 2
        already figured out about the chunk's content type."""
        system, user = prompts.detection_prompt(chunk_text, targets, classification)
        if self._prompt_blank(system, user):
            return None
        out = self._call(self.cfg.cheap, system, user, max_tokens=200)
        return self._parse_detection_result(out)

    # Final-stage (extraction) vocab. Deliberately a different, coarser
    # hierarchy than Stage 3's _MATCH_TYPES: Stage 3 is a recall-biased cheap
    # screen (finer exact/semantic/conceptual/related/no_match), while
    # extraction is the precision-biased final call — it only ever emits a
    # record when a side resolves to exact or synonym, so it doesn't need
    # Stage 3's middle granularity. Mirrors models.MatchType exactly.
    _RECORD_MATCH_TYPES = ("exact", "synonym", "related_separate", "no_match")
    _TRANSFORMATION_TYPES = (
        "none", "log", "log_difference", "ratio", "percent_change",
        "first_difference", "growth_rate", "index", "standardized",
        "other", "unknown",
    )
    _SPEC_STATUSES = ("explicit_baseline", "inferred_baseline", "unknown")
    _UNIT_SOURCES = ("table", "data_section", "inferred", "unknown")
    # Mirrors models.VariableRole/EstimateType/SourceType.
    _VARIABLE_ROLES = ("treatment", "primary_regressor", "control", "unknown")
    _ESTIMATE_TYPES = (
        "regression_coefficient", "elasticity", "semi_elasticity", "multiplier",
        "impulse_response", "marginal_effect", "hazard_ratio", "other",
    )
    _SOURCE_TYPES = ("table", "text", "figure")
    _TEST_STAT_TYPES = ("t", "z", "chi2", "f", "other")

    @classmethod
    def _clean_enum(cls, value: Any, allowed: tuple) -> Optional[str]:
        if not isinstance(value, str):
            return None
        v = value.strip().lower()
        return v if v in allowed else None

    @classmethod
    def _parse_one_estimate(cls, raw: Any) -> Optional[Dict[str, Any]]:
        """Validate/clean a single estimate dict from the model. Numeric and
        free-text fields are passed through as-is (the model is the main,
        expensive tier — trusted more than the cheap stages — but enum fields
        are still validated strictly rather than trusted blindly, same
        defensive-parsing principle as Stages 2/3). Returns None if `raw`
        isn't even a dict, or if it has neither a coefficient nor a real
        elasticity-type/product target match — i.e. nothing worth keeping."""
        if not isinstance(raw, dict):
            return None
        out: Dict[str, Any] = dict(raw)
        out["match_type"] = cls._clean_enum(raw.get("match_type"), cls._RECORD_MATCH_TYPES)
        out["elasticity_transformation_type"] = cls._clean_enum(raw.get("elasticity_transformation_type"), cls._TRANSFORMATION_TYPES)
        out["specification_status"] = cls._clean_enum(raw.get("specification_status"), cls._SPEC_STATUSES)
        out["unit_source"] = cls._clean_enum(raw.get("unit_source"), cls._UNIT_SOURCES)
        out["variable_role"] = cls._clean_enum(raw.get("variable_role"), cls._VARIABLE_ROLES)
        out["estimate_type"] = cls._clean_enum(raw.get("estimate_type"), cls._ESTIMATE_TYPES)
        out["source_type"] = cls._clean_enum(raw.get("source_type"), cls._SOURCE_TYPES)

        # test_statistic: {"type": "t"/"z"/"chi2"/"f"/"other", "value": float}
        # — only kept if it has a usable numeric value; an unrecognized type
        # string falls back to "other" rather than discarding the number.
        ts = raw.get("test_statistic")
        if isinstance(ts, dict) and isinstance(ts.get("value"), (int, float)):
            out["test_statistic"] = {
                "type": cls._clean_enum(ts.get("type"), cls._TEST_STAT_TYPES) or "other",
                "value": ts["value"],
            }
        else:
            out["test_statistic"] = None

        # *_reported flags: honor an explicit bool from the model, otherwise
        # derive from whether the corresponding value is actually present —
        # distinguishes "paper didn't report this" from "extractor found
        # nothing", per the reviewed schema's missing-vs-absent distinction.
        def _reported(value_key: str, flag_key: str) -> bool:
            explicit = raw.get(flag_key)
            if isinstance(explicit, bool):
                return explicit
            return raw.get(value_key) is not None
        out["p_value_reported"] = _reported("p_value", "p_value_reported")
        out["standard_error_reported"] = _reported("standard_error", "standard_error_reported")
        out["confidence_interval_reported"] = _reported("confidence_interval", "confidence_interval_reported")

        # review_reason: tolerate a bare string, coerce to a list either way
        rr = raw.get("review_reason")
        if isinstance(rr, str):
            out["review_reason"] = [rr] if rr.strip() else []
        elif isinstance(rr, list):
            out["review_reason"] = [r for r in rr if isinstance(r, str)]
        else:
            out["review_reason"] = []
        out["requires_review"] = bool(raw.get("requires_review")) if isinstance(raw.get("requires_review"), bool) else bool(out["review_reason"])
        # nothing usable at all: no coefficient AND no target match on either
        # side — the model is instructed to only emit real matches, but a
        # defensive check here costs nothing.
        has_coef = isinstance(raw.get("coefficient"), (int, float))
        has_match = out["match_type"] in ("exact", "synonym")
        if not has_coef and not has_match:
            return None
        return out

    @classmethod
    def _parse_extraction_result(cls, raw: str) -> Optional[list]:
        """Defensive parsing for Stage 4's JSON-structured output:
        {"estimates": [ {...}, ... ]}. A single excerpt (especially a
        multi-row/column regression table) can legitimately produce zero,
        one, or several records, so this always returns a list (possibly
        empty) rather than a single dict. Tolerates a model that forgets the
        "estimates" wrapper and returns one bare estimate object instead."""
        if not raw:
            return None
        parsed = cls._parse_json(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("estimates"), list):
            candidates = parsed["estimates"]
        elif isinstance(parsed, dict):
            # legacy/lenient: model returned a single estimate object with no
            # "estimates" wrapper — treat it as a one-element list.
            candidates = [parsed]
        else:
            return None
        cleaned = [cls._parse_one_estimate(c) for c in candidates]
        return [c for c in cleaned if c is not None]

    # ----- Stage 4a: paper-level metadata (main, once per paper) -----
    @classmethod
    def _parse_paper_metadata_result(cls, raw: str) -> Optional[Dict[str, Any]]:
        """Defensive parsing for the once-per-paper metadata call:
        {"model_type", "countries_region", "time_period": {"start","end"},
        "frequency", "n_obs", "n_units", "data_source"}. Every field is
        optional/nullable — a paper whose methodology chunk didn't state its
        sample size, say, should come back with n_obs: None rather than
        failing the whole call."""
        if not raw:
            return None
        parsed = cls._parse_json(raw)
        if not isinstance(parsed, dict):
            return None
        tp = parsed.get("time_period")
        time_period = None
        if isinstance(tp, dict):
            start = tp.get("start")
            end = tp.get("end")
            time_period = {
                "start": start if isinstance(start, int) else None,
                "end": end if isinstance(end, int) else None,
            }
        return {
            "model_type": parsed.get("model_type") if isinstance(parsed.get("model_type"), str) else None,
            "countries_region": parsed.get("countries_region") if isinstance(parsed.get("countries_region"), str) else None,
            "time_period": time_period,
            "frequency": parsed.get("frequency") if isinstance(parsed.get("frequency"), str) else None,
            "n_obs": parsed.get("n_obs") if isinstance(parsed.get("n_obs"), int) else None,
            "n_units": parsed.get("n_units") if isinstance(parsed.get("n_units"), int) else None,
            "data_source": parsed.get("data_source") if isinstance(parsed.get("data_source"), str) else None,
        }

    def extract_paper_metadata(self, abstract: Optional[str],
                                methodology_text: str) -> Optional[Dict[str, Any]]:
        """One call per PAPER (not per chunk): pulls out the facts that are
        constant across nearly all of that paper's estimates — country/
        region, sample period, frequency, N, data source, primary model/
        method — from the abstract plus every chunk Stage 2 classified as
        methodology or data_description. Returns None if the prompt is blank
        or nothing parseable came back."""
        system, user = prompts.paper_metadata_prompt(abstract, methodology_text)
        if self._prompt_blank(system, user):
            return None
        out = self._call(self.cfg.main, system, user, max_tokens=400)
        return self._parse_paper_metadata_result(out)

    # ----- Stage 4: extraction (main) -----
    def extract(self, chunk_text: str, targets: dict, ontology: dict,
                abstract: Optional[str] = None,
                screening: Optional[Dict[str, Any]] = None) -> Optional[list]:
        """Returns a list of zero or more estimate dicts (see
        _parse_extraction_result), or None if the prompt is blank or nothing
        parseable came back at all.

        `screening` bundles Stage 2's classification AND Stage 3's detection
        output for this same chunk (labels, target_match, keep_reason, etc.)
        — handed over as prior-screening context, same reasoning as Stage 3
        receiving Stage 2's classification."""
        system, user = prompts.extraction_prompt(chunk_text, targets, ontology, abstract, screening)
        if self._prompt_blank(system, user):
            return None
        out = self._call(self.cfg.main, system, user)
        return self._parse_extraction_result(out)

    # ----- Stage 5: validation -----
    def validate(self, record_json: str, chunk_text: str) -> Optional[Dict[str, Any]]:
        system, user = prompts.validation_prompt(record_json, chunk_text)
        if self._prompt_blank(system, user):
            return None
        out = self._call(self.cfg.validation, system, user)
        return self._parse_json(out)
