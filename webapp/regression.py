"""Meta-regression over extracted elasticity records — the final analysis
step: define a dependent variable and independent variables from the
extracted dataset, fit an OLS (or precision-weighted WLS) model, and format
the result as an economics-paper-style table — coefficients with
significance stars and standard errors, plus the usual model-fit statistics
(N, R-squared, F-statistic).

Kept as a webapp-only concern (not part of meta_pipeline/) since it operates
on the already-extracted records table as exploratory analysis, rather than
being a pipeline stage that produces new per-paper output.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

# Which record fields can be used as a DV or IV, and how they're treated in
# the regression formula. "derived" fields are computed from raw record
# fields (e.g. abs_coefficient from coefficient) rather than being a literal
# key already present on the record dict — see _row_from_record.
# Standard 8-group food classification (mirrors meta_pipeline.llm.LLMClient
# ._FOOD_GROUPS / models.FoodGroup exactly) — duplicated here as a small,
# stable constant rather than importing meta_pipeline.llm, which would pull
# in the anthropic/openai/google SDKs just for an 8-item tuple.
FOOD_GROUPS = (
    "Bread and cereals", "Meat", "Fish and seafood", "Dairy products",
    "Fats and oils", "Fruit and vegetables", "Beverages and tobacco",
    "Other food products",
)
FIELDS: Dict[str, Dict[str, Any]] = {
    "coefficient":                 {"label": "Coefficient (elasticity value)", "kind": "numeric"},
    "abs_coefficient":             {"label": "|Coefficient| (magnitude)", "kind": "numeric", "derived": True},
    "standard_error":              {"label": "Standard error", "kind": "numeric"},
    "p_value":                     {"label": "P-value", "kind": "numeric"},
    "n_obs":                       {"label": "N (observations)", "kind": "numeric"},
    "n_units":                     {"label": "N (units)", "kind": "numeric"},
    "time_period_midpoint":        {"label": "Sample midpoint year", "kind": "numeric", "derived": True},
    "time_period_span":            {"label": "Sample period length (yrs)", "kind": "numeric", "derived": True},
    "target_elasticity_type":      {"label": "Elasticity type", "kind": "categorical"},
    "target_product":              {"label": "Product", "kind": "categorical"},
    "target_cross_price_product":  {"label": "Cross-price product", "kind": "categorical"},
    "target_food_group":           {"label": "Food group", "kind": "categorical", "options": list(FOOD_GROUPS)},
    "target_cross_price_food_group": {"label": "Cross-price food group", "kind": "categorical", "options": list(FOOD_GROUPS)},
    "match_type":                  {"label": "Match type", "kind": "categorical"},
    "variable_role":               {"label": "Variable role", "kind": "categorical"},
    "estimate_type":                {"label": "Estimate type", "kind": "categorical"},
    "specification_status":        {"label": "Specification status", "kind": "categorical"},
    "unit_source":                 {"label": "Unit source", "kind": "categorical"},
    "source_type":                 {"label": "Source type", "kind": "categorical"},
    "model_type":                  {"label": "Model type", "kind": "categorical"},
    "countries_region":            {"label": "Country / region", "kind": "categorical"},
    "frequency":                   {"label": "Data frequency", "kind": "categorical"},
    "data_source":                 {"label": "Data source", "kind": "categorical"},
    "paper_id":                    {"label": "Paper (fixed effect)", "kind": "categorical"},
    "elasticity_is_raw":           {"label": "Elasticity is raw (vs. derived)", "kind": "boolean"},
    "requires_review":             {"label": "Flagged for review", "kind": "boolean"},
    "manually_edited":             {"label": "Manually edited", "kind": "boolean"},
}

DEFAULT_DV = "coefficient"


class RegressionError(ValueError):
    """User-facing error — bad field choice, not enough data, a model that
    failed to fit, etc. The route catches this and returns a 400 with the
    message as-is, rather than a raw traceback."""


def field_catalog() -> Dict[str, Any]:
    return {"fields": FIELDS, "default_dv": DEFAULT_DV}


def _safe_float(v: Any) -> Optional[float]:
    """float(v), but None (not NaN) for anything that isn't a real finite
    number — a bare NaN in the JSON response would break JSON.parse on the
    frontend (unlike Python's json module, strict JSON has no NaN literal)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN != NaN is the classic NaN check


def _row_from_record(r: dict) -> dict:
    row: Dict[str, Any] = {}
    coef = r.get("coefficient")
    row["coefficient"] = coef if isinstance(coef, (int, float)) else None
    row["abs_coefficient"] = abs(coef) if isinstance(coef, (int, float)) else None
    row["standard_error"] = r.get("standard_error")
    row["p_value"] = r.get("p_value")
    row["n_obs"] = r.get("n_obs")
    row["n_units"] = r.get("n_units")

    tp = r.get("time_period") or {}
    start, end = tp.get("start"), tp.get("end")
    if start is not None and end is not None:
        row["time_period_midpoint"] = (start + end) / 2
        row["time_period_span"] = end - start
    elif start is not None or end is not None:
        row["time_period_midpoint"] = start if start is not None else end
        row["time_period_span"] = None
    else:
        row["time_period_midpoint"] = None
        row["time_period_span"] = None

    for f in ("target_elasticity_type", "target_product", "target_cross_price_product",
              "target_food_group", "target_cross_price_food_group",
              "match_type", "variable_role", "estimate_type", "specification_status",
              "unit_source", "source_type", "model_type", "countries_region",
              "frequency", "data_source", "paper_id"):
        row[f] = r.get(f)

    row["elasticity_is_raw"] = r.get("elasticity_is_raw")
    row["requires_review"] = r.get("requires_review")
    row["manually_edited"] = bool(r.get("manually_edited"))
    return row


def build_dataframe(records: List[dict], filters: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """One row per extraction record, with every FIELDS entry computed.
    `filters` optionally restricts to records where a given field's raw
    value is in an allowed set (e.g. {"target_product": ["Maize", "Wheat"]})
    — applied before derived-field computation, since filters always target
    one of the actual record fields, not a derived one."""
    filters = filters or {}
    rows = []
    for r in records:
        ok = True
        for field, allowed in filters.items():
            if not allowed:
                continue
            if r.get(field) not in allowed:
                ok = False
                break
        if ok:
            rows.append(_row_from_record(r))
    return pd.DataFrame(rows, columns=list(FIELDS.keys()))


def _stars(p: Optional[float]) -> str:
    if p is None or p != p:  # NaN check
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


_DUMMY_RE = re.compile(r"^C\((\w+)\)\[T\.(.+)\]$")


def _clean_term_name(name: str) -> str:
    if name == "Intercept":
        return "Intercept"
    m = _DUMMY_RE.match(name)
    if m:
        field, level = m.group(1), m.group(2)
        label = FIELDS.get(field, {}).get("label", field)
        return f"{label}: {level}"
    return FIELDS.get(name, {}).get("label", name)


def run_regression(records: List[dict], dv: str, ivs: List[str],
                    weight_by_precision: bool = False,
                    filters: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """Fit dv ~ ivs (OLS, or precision-weighted WLS with weight 1/SE^2 when
    weight_by_precision is set — a standard meta-regression technique, since
    a more precisely estimated elasticity should count for more than a
    loosely estimated one). Categorical/boolean IVs are dummy-coded
    automatically via statsmodels' formula API (C(field)), with the first
    observed level dropped as the reference category — same convention as a
    standard econometrics regression table.

    Returns one row per coefficient (variable name, coefficient, standard
    error, t-stat, p-value, significance stars, 95% CI) plus model-level
    statistics (N, R-squared, Adjusted R-squared, F-statistic and its
    p-value, residual/model degrees of freedom) — the same information a
    published regression table reports.
    """
    import statsmodels.formula.api as smf

    if dv not in FIELDS:
        raise RegressionError(f"Unknown dependent variable '{dv}'")
    for f in ivs:
        if f not in FIELDS:
            raise RegressionError(f"Unknown independent variable '{f}'")
    if dv in ivs:
        raise RegressionError("The dependent variable can't also be an independent variable")

    df = build_dataframe(records, filters)
    if df.empty:
        raise RegressionError("No extracted records available to regress on")

    needed = [dv] + ivs
    if weight_by_precision:
        needed.append("standard_error")
    clean = df.dropna(subset=needed).copy()
    if weight_by_precision:
        clean = clean[clean["standard_error"] > 0]
    if len(clean) < len(ivs) + 2:
        raise RegressionError(
            f"Only {len(clean)} complete observation(s) available for a model with "
            f"{len(ivs)} independent variable(s) plus an intercept — try dropping an "
            f"IV, picking one with fewer missing values, or loosening any filters."
        )

    terms = [f"C({f})" if FIELDS[f]["kind"] in ("categorical", "boolean") else f for f in ivs]
    formula = f"{dv} ~ " + " + ".join(terms) if terms else f"{dv} ~ 1"

    try:
        if weight_by_precision:
            weights = 1.0 / (clean["standard_error"] ** 2)
            model = smf.wls(formula=formula, data=clean, weights=weights).fit()
        else:
            model = smf.ols(formula=formula, data=clean).fit()
    except Exception as e:
        raise RegressionError(f"Regression failed to fit: {e}") from e

    ci = model.conf_int()
    rows = []
    for name in model.params.index:
        p = _safe_float(model.pvalues[name])
        rows.append({
            "variable": _clean_term_name(name),
            "coefficient": _safe_float(model.params[name]),
            "std_error": _safe_float(model.bse[name]),
            "t_stat": _safe_float(model.tvalues[name]),
            "p_value": p,
            "stars": _stars(p),
            "ci_low": _safe_float(ci.loc[name, 0]),
            "ci_high": _safe_float(ci.loc[name, 1]),
        })

    return {
        "formula": formula,
        "dv": dv,
        "ivs": ivs,
        "weight_by_precision": weight_by_precision,
        "filters": filters or {},
        "rows": rows,
        "n_obs": int(model.nobs),
        "r_squared": _safe_float(model.rsquared),
        "adj_r_squared": _safe_float(model.rsquared_adj),
        "f_statistic": _safe_float(getattr(model, "fvalue", None)),
        "f_pvalue": _safe_float(getattr(model, "f_pvalue", None)),
        "df_model": _safe_float(model.df_model),
        "df_resid": _safe_float(model.df_resid),
    }
