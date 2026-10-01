"""Chen et al.–style meta-analysis tables for the Dashboard.

Builds Tables 1–4 from extracted records:
  1. Descriptive elasticity stats by product group (income + Marshallian own-price)
  2. Sample characteristics (% of estimates) + mean real income
  3. Meta-regression (product dummies × ln income + study covariates) with
     Chen variance-equation WLS weighting
  4. Predicted elasticities at the filtered sample's mean ln(income)

Full §3 conversions (conditional→unconditional, Slutsky, symmetry) are not
applied yet — tables use extracted coefficients as reported, remapped onto
Chen's nine product groups. Rows without a usable coefficient are dropped;
Table 3/4 rows without real_per_capita_income are excluded from regressions
that need ln(Y) (descriptives still include them where possible).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

FOOD_GROUPS = (
    "Grains and vegetables", "Meat and eggs", "Edible oil",
    "Aquatic products", "Fruits", "Sugar", "Dairy", "Tobacco", "Alcohol",
)

# Map legacy 8-way labels still sitting on older records.json rows.
_LEGACY_FOOD_GROUP = {
    "Bread and cereals": "Grains and vegetables",
    "Meat": "Meat and eggs",
    "Fish and seafood": "Aquatic products",
    "Dairy products": "Dairy",
    "Fats and oils": "Edible oil",
    "Fruit and vegetables": "Grains and vegetables",
    "Beverages and tobacco": "Alcohol",
    "Other food products": "Sugar",
}

_INCOME_RE = re.compile(r"income|expenditure", re.I)
_OWN_RE = re.compile(r"own[\s\-]?price", re.I)
_CROSS_RE = re.compile(r"cross[\s\-]?price", re.I)


def normalize_food_group(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    if v in FOOD_GROUPS:
        return v
    mapped = _LEGACY_FOOD_GROUP.get(v)
    if mapped:
        return mapped
    lower = v.lower()
    for g in FOOD_GROUPS:
        if g.lower() == lower:
            return g
    return _LEGACY_FOOD_GROUP.get(next((k for k in _LEGACY_FOOD_GROUP if k.lower() == lower), ""), None) or None


def classify_elasticity(rec: dict) -> Optional[str]:
    """Return 'income' | 'own_price' | 'cross_price' | None."""
    t = rec.get("target_elasticity_type") or ""
    if _CROSS_RE.search(t):
        return "cross_price"
    if _OWN_RE.search(t):
        return "own_price"
    if _INCOME_RE.search(t):
        return "income"
    # Fall back on presence of cross-price product
    if rec.get("target_cross_price_product") or rec.get("target_cross_price_food_group"):
        return "cross_price"
    return None


def _safe_float(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or math.isinf(f):
        return None
    return f


def _country_key(raw: Any) -> Optional[str]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def list_countries(records: List[dict]) -> List[str]:
    counts = Counter(_country_key(r.get("countries_region")) for r in records)
    return [c for c, _ in counts.most_common() if c]


def filter_records(records: List[dict], countries: Optional[List[str]] = None) -> List[dict]:
    if not countries:
        return list(records)
    allowed = set(countries)
    return [r for r in records if _country_key(r.get("countries_region")) in allowed]


def _prep_row(rec: dict) -> Optional[dict]:
    coef = _safe_float(rec.get("coefficient"))
    if coef is None:
        return None
    kind = classify_elasticity(rec)
    if kind is None:
        return None
    group = normalize_food_group(rec.get("target_food_group"))
    income = _safe_float(rec.get("real_per_capita_income"))
    n_obs = _safe_float(rec.get("n_obs"))
    n_j = _safe_float(rec.get("n_products_in_demand_system"))
    return {
        "coefficient": coef,
        "kind": kind,
        "food_group": group,
        "income": income,
        "ln_income": math.log(income) if income is not None and income > 0 else None,
        "n_obs": n_obs if n_obs and n_obs > 0 else None,
        "n_j": n_j if n_j and n_j > 0 else None,
        "ln_n": math.log(n_obs) if n_obs and n_obs > 0 else None,
        "ln_j": math.log(n_j) if n_j and n_j > 0 else None,
        "data_level": rec.get("data_level"),
        "geographic_scope": rec.get("geographic_scope"),
        "urban_rural": rec.get("urban_rural"),
        "data_type": rec.get("data_type"),
        "price_measure": rec.get("price_measure"),
        "conditioning": rec.get("conditioning") or "unconditional",
        "budgeting_stages": rec.get("budgeting_stages"),
        "demand_system": rec.get("demand_system"),
        "estimation_method": rec.get("estimation_method"),
        "demographic_controls": rec.get("demographic_controls"),
        "publication_status": rec.get("publication_status") or "unpublished",
        "study_language": rec.get("study_language") or "english",
        "country": _country_key(rec.get("countries_region")),
        "paper_id": rec.get("paper_id"),
        "requires_review": bool(rec.get("requires_review")),
    }


def prepare_sample(records: List[dict], countries: Optional[List[str]] = None) -> List[dict]:
    return [row for r in filter_records(records, countries) if (row := _prep_row(r))]


def _desc(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = None
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": min(vals),
        "max": max(vals),
    }


def table1_summary(rows: List[dict]) -> Dict[str, Any]:
    """By product group: n/mean/sd/min/max for income and own-price elasticities."""
    by_group: Dict[str, Dict[str, List[float]]] = {
        g: {"income": [], "own_price": []} for g in FOOD_GROUPS
    }
    ungrouped = {"income": [], "own_price": []}
    for r in rows:
        bucket = by_group.get(r["food_group"]) if r["food_group"] else None
        if r["kind"] == "income":
            (bucket or ungrouped)["income"].append(r["coefficient"])
        elif r["kind"] == "own_price":
            (bucket or ungrouped)["own_price"].append(r["coefficient"])

    products = []
    for g in FOOD_GROUPS:
        products.append({
            "product": g,
            "income": _desc(by_group[g]["income"]),
            "own_price": _desc(by_group[g]["own_price"]),
        })
    if ungrouped["income"] or ungrouped["own_price"]:
        products.append({
            "product": "Ungrouped / other",
            "income": _desc(ungrouped["income"]),
            "own_price": _desc(ungrouped["own_price"]),
        })
    return {
        "products": products,
        "n_income": sum(1 for r in rows if r["kind"] == "income"),
        "n_own_price": sum(1 for r in rows if r["kind"] == "own_price"),
    }


_TABLE2_SPECS = [
    ("data_level", "Data level", {
        "household_individual": "Household / individual",
        "aggregate": "Aggregate",
    }),
    ("geographic_scope", "Geographic scope", {
        "national": "National",
        "regional": "Regional",
    }),
    ("urban_rural", "Urban / rural", {
        "both": "Both",
        "urban_only": "Urban only",
        "rural_only": "Rural only",
    }),
    ("data_type", "Data type", {
        "cross_section": "Cross-section",
        "time_series": "Time series",
        "pooled": "Pooled",
        "panel": "Panel",
    }),
    ("price_measure", "Price measure", {
        "actual_prices": "Actual prices",
        "unit_values": "Unit values",
    }),
    ("conditioning", "Conditioning", {
        "unconditional": "Unconditional",
        "conditional_food": "Conditional (food)",
        "conditional_animal": "Conditional (animal)",
        "conditional_grain": "Conditional (grain)",
    }),
    ("budgeting_stages", "Budgeting stages", {
        "single_stage": "Single-stage",
        "multi_stage": "Multi-stage",
    }),
    ("demand_system", "Demand system", {
        "none": "None / double-log",
        "les_qes": "LES / QES",
        "aids": "AIDS",
        "quaids": "QUAIDS",
        "translog": "Translog",
        "linquad": "LinQuad",
        "rotterdam": "Rotterdam",
    }),
    ("estimation_method", "Estimation method", {
        "ols": "OLS",
        "ml": "ML",
        "sur": "SUR",
        "gls": "GLS",
        "other": "Other",
    }),
    ("demographic_controls", "Demographic controls", {
        True: "Yes",
        False: "No",
    }),
    ("publication_status", "Publication status", {
        "published": "Published",
        "unpublished": "Unpublished",
    }),
    ("study_language", "Study language", {
        "english": "English",
        "other": "Other",
    }),
]


def table2_characteristics(rows: List[dict]) -> Dict[str, Any]:
    n = len(rows) or 1
    incomes = [r["income"] for r in rows if r["income"] is not None]
    sections = []
    for field, title, labels in _TABLE2_SPECS:
        counts: Counter = Counter()
        missing = 0
        for r in rows:
            v = r.get(field)
            if v is None or v == "":
                missing += 1
            else:
                counts[v] += 1
        items = []
        for key, label in labels.items():
            c = counts.get(key, 0)
            items.append({
                "label": label,
                "count": c,
                "pct": 100.0 * c / n if rows else 0.0,
            })
        if missing:
            items.append({
                "label": "Missing / not extracted",
                "count": missing,
                "pct": 100.0 * missing / n if rows else 0.0,
            })
        sections.append({"title": title, "items": items})

    return {
        "n_estimates": len(rows),
        "real_income": _desc(incomes),
        "n_with_income": len(incomes),
        "sections": sections,
    }


def _chen_stars(p: Optional[float]) -> str:
    if p is None or p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""


def _dummy_cols(series_vals: List[Any], prefix: str, drop_first: bool = True) -> Tuple[List[str], Dict[str, List[float]]]:
    levels = []
    seen = set()
    for v in series_vals:
        if v is None or v == "":
            continue
        if v not in seen:
            seen.add(v)
            levels.append(v)
    if drop_first and levels:
        levels = levels[1:]
    cols = {}
    names = []
    for lvl in levels:
        name = f"{prefix}[{lvl}]"
        names.append(name)
        cols[name] = [1.0 if v == lvl else 0.0 for v in series_vals]
    return names, cols


def _fit_chen_equation(sample: List[dict], kind: str) -> Dict[str, Any]:
    """Fit one Chen-style equation for income / own_price / cross_price."""
    try:
        import numpy as np
        import statsmodels.api as sm
    except ImportError as e:
        return {
            "ok": False,
            "error": "Install pandas/numpy/statsmodels (see requirements.txt) to run Table 3.",
            "detail": str(e),
            "kind": kind,
        }

    rows = [r for r in sample if r["kind"] == kind and r["food_group"]]
    notes = []
    use_income = True
    with_income = [r for r in rows if r["ln_income"] is not None]
    if len(with_income) < 8:
        use_income = False
        notes.append(
            "Few rows have real_per_capita_income — fitting product dummies only "
            "(no ln(Y)×group interactions). Re-extract papers to fill income."
        )
    else:
        rows = with_income
        notes.append(f"Using {len(rows)} rows with real per capita income for ln(Y) interactions.")

    if len(rows) < 6:
        return {
            "ok": False,
            "error": f"Not enough {kind.replace('_', '-')} estimates after filters (n={len(rows)}).",
            "kind": kind,
            "n_obs": len(rows),
            "notes": notes,
        }

    y = np.array([r["coefficient"] for r in rows], dtype=float)
    groups = [r["food_group"] for r in rows]
    # Reference = Grains and vegetables when present, else first observed
    ref = "Grains and vegetables" if "Grains and vegetables" in groups else sorted(set(groups))[0]
    group_levels = [g for g in FOOD_GROUPS if g in set(groups) and g != ref]
    # Also include any unexpected groups
    for g in sorted(set(groups)):
        if g != ref and g not in group_levels:
            group_levels.append(g)

    X_parts: Dict[str, List[float]] = {}
    names: List[str] = []
    for g in group_levels:
        name = f"Product[{g}]"
        names.append(name)
        X_parts[name] = [1.0 if r["food_group"] == g else 0.0 for r in rows]

    if use_income:
        # lnY main effect + interactions with non-reference products
        names.append("ln(Y)")
        X_parts["ln(Y)"] = [r["ln_income"] for r in rows]
        for g in group_levels:
            name = f"ln(Y)×{g}"
            names.append(name)
            X_parts[name] = [
                (r["ln_income"] if r["food_group"] == g else 0.0) for r in rows
            ]

    # Study covariates (reference categories dropped)
    cov_specs = [
        ("data_level", "DataLevel", ["household_individual"]),
        ("geographic_scope", "Geo", ["national"]),
        ("urban_rural", "UrbanRural", ["both"]),
        ("data_type", "DataType", ["cross_section"]),
        ("price_measure", "Prices", ["actual_prices"]),
        ("conditioning", "Cond", ["unconditional"]),
        ("budgeting_stages", "Budget", ["single_stage"]),
        ("demand_system", "DemandSys", ["none"]),
        ("estimation_method", "Est", ["ols"]),
        ("publication_status", "Pub", ["unpublished"]),
        ("study_language", "Lang", ["english"]),
    ]
    for field, prefix, drop in cov_specs:
        vals = [r.get(field) for r in rows]
        levels = []
        for v in vals:
            if v is None or v == "" or v in drop:
                continue
            if v not in levels:
                levels.append(v)
        for lvl in levels:
            name = f"{prefix}[{lvl}]"
            names.append(name)
            X_parts[name] = [1.0 if v == lvl else 0.0 for v in vals]

    dem = [r.get("demographic_controls") for r in rows]
    if any(v is True for v in dem):
        names.append("Demographics")
        X_parts["Demographics"] = [1.0 if v is True else 0.0 for v in dem]

    if not names:
        return {
            "ok": False,
            "error": "No regressors available after coding.",
            "kind": kind,
            "n_obs": len(rows),
            "notes": notes,
        }

    X = np.column_stack([np.ones(len(rows))] + [np.array(X_parts[n], dtype=float) for n in names])
    full_names = ["Intercept"] + names

    # Drop collinear / near-zero-variance columns
    keep = [0]  # always keep intercept
    for j in range(1, X.shape[1]):
        col = X[:, j]
        if float(np.std(col)) < 1e-12:
            continue
        # check rank if added
        trial = X[:, keep + [j]]
        if np.linalg.matrix_rank(trial, tol=1e-8) > len(keep):
            keep.append(j)
    dropped = [full_names[j] for j in range(len(full_names)) if j not in keep]
    if dropped:
        notes.append(f"Dropped collinear/constant columns: {', '.join(dropped[:8])}"
                     + ("…" if len(dropped) > 8 else ""))
    X = X[:, keep]
    full_names = [full_names[j] for j in keep]

    if X.shape[1] >= len(rows):
        return {
            "ok": False,
            "error": f"Too many regressors ({X.shape[1]}) for n={len(rows)}.",
            "kind": kind,
            "n_obs": len(rows),
            "notes": notes,
        }

    # Step 1: OLS
    ols = sm.OLS(y, X).fit()
    resid = ols.resid

    # Step 2: variance equation ln(u²) ~ ln N + ln J + Pub + Lang
    u2 = np.clip(resid ** 2, 1e-12, None)
    ln_u2 = np.log(u2)
    Z_cols = [np.ones(len(rows))]
    z_names = ["var_intercept"]
    ln_n = np.array([r["ln_n"] if r["ln_n"] is not None else np.nan for r in rows])
    ln_j = np.array([r["ln_j"] if r["ln_j"] is not None else np.nan for r in rows])
    pub = np.array([1.0 if r.get("publication_status") == "published" else 0.0 for r in rows])
    lang = np.array([1.0 if r.get("study_language") == "other" else 0.0 for r in rows])

    if np.isfinite(ln_n).sum() >= 5:
        # impute missing ln_n with sample mean of observed
        fill = float(np.nanmean(ln_n))
        ln_n = np.where(np.isfinite(ln_n), ln_n, fill)
        Z_cols.append(ln_n)
        z_names.append("ln(N)")
    else:
        notes.append("Variance equation: ln(N) omitted (insufficient n_obs).")

    if np.isfinite(ln_j).sum() >= 5:
        fill = float(np.nanmean(ln_j))
        ln_j = np.where(np.isfinite(ln_j), ln_j, fill)
        Z_cols.append(ln_j)
        z_names.append("ln(J)")
    else:
        notes.append("Variance equation: ln(J) omitted (insufficient n_products_in_demand_system).")

    if pub.sum() >= 1 and pub.sum() < len(rows):
        Z_cols.append(pub)
        z_names.append("Published")
    if lang.sum() >= 1 and lang.sum() < len(rows):
        Z_cols.append(lang)
        z_names.append("Language[other]")

    Z = np.column_stack(Z_cols)
    # Drop singular / constant Z columns beyond the intercept
    z_keep = [0]
    for j in range(1, Z.shape[1]):
        col = Z[:, j]
        if float(np.std(col)) < 1e-12:
            continue
        trial = Z[:, z_keep + [j]]
        if np.linalg.matrix_rank(trial, tol=1e-8) > len(z_keep):
            z_keep.append(j)
    Z = Z[:, z_keep]
    z_names = [z_names[j] for j in z_keep]

    try:
        var_model = sm.OLS(ln_u2, Z).fit()
        sigma2_hat = np.exp(var_model.fittedvalues)
        sigma2_hat = np.clip(sigma2_hat, 1e-10, None)
        weights = 1.0 / sigma2_hat
        wls = sm.WLS(y, X, weights=weights).fit()
        model = wls
        weighting = "variance_equation"
        var_rows = []
        for name, coef, se, t, p in zip(
            z_names, var_model.params, var_model.bse, var_model.tvalues, var_model.pvalues
        ):
            var_rows.append({
                "variable": name,
                "coefficient": float(coef),
                "std_error": float(se),
                "t": float(t),
                "p": float(p),
                "stars": _chen_stars(float(p)),
            })
    except Exception as e:
        notes.append(f"Variance equation failed ({e}); falling back to OLS.")
        model = ols
        weighting = "ols_fallback"
        var_rows = []

    # Presentation: hide intercept + bare product dummies when ln(Y)
    # interactions are shown (Chen dashboard convention). If we couldn't
    # estimate interactions, keep product dummies visible so the table isn't empty.
    hide_prefixes = ("Intercept",)
    if use_income:
        hide_prefixes = ("Intercept", "Product[")
    display_rows = []
    all_rows = []
    for name, coef, se, t, p in zip(
        full_names, model.params, model.bse, model.tvalues, model.pvalues
    ):
        entry = {
            "variable": name,
            "coefficient": float(coef),
            "std_error": float(se),
            "t": float(t),
            "p": float(p),
            "stars": _chen_stars(float(p)),
            "hidden": name.startswith(hide_prefixes),
        }
        all_rows.append(entry)
        if not entry["hidden"]:
            display_rows.append(entry)

    return {
        "ok": True,
        "kind": kind,
        "label": {"income": "Income", "own_price": "Own-price", "cross_price": "Cross-price"}[kind],
        "n_obs": int(len(rows)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(getattr(model, "rsquared_adj", model.rsquared)),
        "weighting": weighting,
        "reference_product": ref,
        "notes": notes,
        "rows": display_rows,
        "all_rows": all_rows,
        "variance_equation": var_rows,
        "params": {r["variable"]: r["coefficient"] for r in all_rows},
        "use_income": use_income,
        "mean_ln_income": float(np.mean([r["ln_income"] for r in rows if r["ln_income"] is not None]))
        if use_income else None,
    }


def table3_regression(rows: List[dict]) -> Dict[str, Any]:
    equations = {}
    for kind in ("income", "own_price", "cross_price"):
        equations[kind] = _fit_chen_equation(rows, kind)
    return {"equations": equations}


def table4_predictions(rows: List[dict], table3: Dict[str, Any],
                       ln_income: Optional[float] = None) -> Dict[str, Any]:
    """Predict income & own-price elasticities at mean (or provided) ln(Y)."""
    eqs = table3.get("equations") or {}
    income_eq = eqs.get("income") or {}
    own_eq = eqs.get("own_price") or {}

    if ln_income is None:
        candidates = [
            eq.get("mean_ln_income")
            for eq in (income_eq, own_eq)
            if eq.get("ok") and eq.get("mean_ln_income") is not None
        ]
        if candidates:
            ln_income = sum(candidates) / len(candidates)
        else:
            incomes = [r["ln_income"] for r in rows if r["ln_income"] is not None]
            ln_income = (sum(incomes) / len(incomes)) if incomes else None

    def predict(eq: dict, group: str) -> Optional[float]:
        if not eq.get("ok"):
            return None
        params = eq.get("params") or {}
        ref = eq.get("reference_product")
        val = params.get("Intercept", 0.0)
        if group != ref:
            val += params.get(f"Product[{group}]", 0.0)
        if eq.get("use_income") and ln_income is not None:
            val += params.get("ln(Y)", 0.0) * ln_income
            if group != ref:
                val += params.get(f"ln(Y)×{group}", 0.0) * ln_income
        return float(val)

    products = []
    for g in FOOD_GROUPS:
        products.append({
            "product": g,
            "income_elasticity": predict(income_eq, g),
            "own_price_elasticity": predict(own_eq, g),
        })

    # Cross-price: mean predicted effect isn't a full matrix without pair dummies;
    # report overall mean cross-price coefficient at sample lnY as a simple summary.
    cross_eq = eqs.get("cross_price") or {}
    cross_note = None
    if cross_eq.get("ok"):
        cross_note = (
            "Cross-price Table 4 matrix needs pair-level dummies; showing product-level "
            "own-price / income predictions only for now."
        )

    return {
        "ln_income": ln_income,
        "income_level": math.exp(ln_income) if ln_income is not None else None,
        "products": products,
        "notes": [n for n in [
            None if income_eq.get("ok") else "Income equation not available for predictions.",
            None if own_eq.get("ok") else "Own-price equation not available for predictions.",
            cross_note,
            None if ln_income is not None else "No ln(Y) available — predictions need real_per_capita_income.",
        ] if n],
    }


def build_dashboard(records: List[dict], countries: Optional[List[str]] = None) -> Dict[str, Any]:
    all_countries = list_countries(records)
    rows = prepare_sample(records, countries)
    t1 = table1_summary(rows)
    t2 = table2_characteristics(rows)
    t3 = table3_regression(rows)
    t4 = table4_predictions(rows, t3)

    n_review = sum(1 for r in rows if r["requires_review"])
    n_income_missing = sum(1 for r in rows if r["income"] is None)

    return {
        "countries_available": all_countries,
        "countries_filter": countries or [],
        "n_records_raw": len(records),
        "n_estimates": len(rows),
        "n_requires_review": n_review,
        "n_missing_income": n_income_missing,
        "table1": t1,
        "table2": t2,
        "table3": t3,
        "table4": t4,
    }
