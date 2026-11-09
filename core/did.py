"""
Core Causal Inference & Experimentation Platform Engine
Module: Difference-in-Differences (DiD) & Dynamic Event Study
"""

from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def estimate_did(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    post_col: Optional[str] = None,
    cutoff_time: Optional[Any] = None,
    covariates: Optional[List[str]] = None,
    cov_type: str = "HC3",
) -> Dict[str, Any]:
    """
    Fits standard Difference-in-Differences (DiD) regression using Two-Way OLS.

    Model:
        Outcome = beta_0 + beta_1 * Treat + beta_2 * Post + beta_did * (Treat * Post) + gamma * Covariates + eps

    Args:
        df: DataFrame containing panel data
        unit_col: Column identifying region / market / unit
        time_col: Column identifying time / date / week
        treatment_col: Binary indicator (1 if ever treated, 0 if control unit)
        outcome_col: Dependent variable / metric
        post_col: Binary indicator (1 if post-treatment period, 0 if pre-period)
        cutoff_time: If post_col is not provided, time >= cutoff_time will define post period
        covariates: Optional list of additional control variable names
        cov_type: Robust standard error covariance type ('HC1', 'HC3', 'cluster')

    Returns:
        Dict with point estimate, standard errors, p-values, 2x2 means table, and model summary.
    """
    data = df.copy()

    # Define post indicator if cutoff_time provided
    if post_col is None:
        if cutoff_time is None:
            raise ValueError("Either 'post_col' or 'cutoff_time' must be specified.")
        data["post"] = (data[time_col] >= cutoff_time).astype(int)
        post_col = "post"

    # Ensure binary indicators
    data["treat_binary"] = data[treatment_col].astype(int)
    data["post_binary"] = data[post_col].astype(int)
    data["did_interaction"] = data["treat_binary"] * data["post_binary"]

    # Compute 2x2 empirical means
    means_2x2 = (
        data.groupby(["treat_binary", "post_binary"])[outcome_col]
        .mean()
        .unstack(fill_value=np.nan)
    )

    ctrl_pre = float(means_2x2.loc[0, 0]) if 0 in means_2x2.index and 0 in means_2x2.columns else np.nan
    ctrl_post = float(means_2x2.loc[0, 1]) if 0 in means_2x2.index and 1 in means_2x2.columns else np.nan
    treat_pre = float(means_2x2.loc[1, 0]) if 1 in means_2x2.index and 0 in means_2x2.columns else np.nan
    treat_post = float(means_2x2.loc[1, 1]) if 1 in means_2x2.index and 1 in means_2x2.columns else np.nan

    delta_ctrl = ctrl_post - ctrl_pre if not (np.isnan(ctrl_post) or np.isnan(ctrl_pre)) else np.nan
    delta_treat = treat_post - treat_pre if not (np.isnan(treat_post) or np.isnan(treat_pre)) else np.nan
    empirical_did = delta_treat - delta_ctrl if not (np.isnan(delta_treat) or np.isnan(delta_ctrl)) else np.nan

    # Build OLS formula
    formula = f"{outcome_col} ~ treat_binary + post_binary + did_interaction"
    if covariates:
        valid_covs = [c for c in covariates if c in data.columns and c not in [outcome_col, treatment_col, post_col]]
        if valid_covs:
            formula += " + " + " + ".join(valid_covs)

    # Fit regression with robust standard errors
    if cov_type == "cluster" and unit_col in data.columns:
        model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[unit_col]})
    else:
        model = smf.ols(formula, data=data).fit(cov_type=cov_type)

    beta_did = float(model.params.get("did_interaction", empirical_did))
    se_did = float(model.bse.get("did_interaction", 0.0))
    t_did = float(model.tvalues.get("did_interaction", 0.0))
    p_did = float(model.pvalues.get("did_interaction", 1.0))
    conf_int = model.conf_int().loc["did_interaction"].values if "did_interaction" in model.conf_int().index else [0.0, 0.0]

    pct_lift = (beta_did / treat_pre) * 100.0 if treat_pre and treat_pre != 0 else 0.0

    # Build time-series trajectories for plotting
    trajectories = (
        data.groupby([time_col, "treat_binary"])[outcome_col]
        .mean()
        .unstack(fill_value=np.nan)
        .reset_index()
    )
    trajectories.columns = [str(c) for c in trajectories.columns]
    # Rename columns to Control (0) and Treated (1)
    if "0" in trajectories.columns:
        trajectories = trajectories.rename(columns={"0": "Control"})
    if "1" in trajectories.columns:
        trajectories = trajectories.rename(columns={"1": "Treated"})

    # Parallel trends counterfactual trajectory for post-period
    if "Control" in trajectories.columns and "Treated" in trajectories.columns:
        trajectories["Counterfactual"] = np.nan
        # Determine pre/post for each row
        if cutoff_time is not None:
            pre_mask = trajectories[time_col] < cutoff_time
            post_mask = trajectories[time_col] >= cutoff_time
        else:
            pre_mask = data.groupby(time_col)["post_binary"].first().values == 0
            post_mask = ~pre_mask

        ctrl_pre_mean = trajectories.loc[pre_mask, "Control"].mean()
        treat_pre_mean = trajectories.loc[pre_mask, "Treated"].mean()
        baseline_offset = treat_pre_mean - ctrl_pre_mean

        # Counterfactual: what treated would have been if it followed control trend
        trajectories.loc[post_mask, "Counterfactual"] = trajectories.loc[post_mask, "Control"] + baseline_offset

    return {
        "did_coefficient": round(beta_did, 4),
        "std_error": round(se_did, 4),
        "t_stat": round(t_did, 3),
        "p_value": float(p_did),
        "ci_lower": round(float(conf_int[0]), 4),
        "ci_upper": round(float(conf_int[1]), 4),
        "pct_lift": round(pct_lift, 2),
        "is_significant": bool(p_did < 0.05),
        "r_squared": round(float(model.rsquared), 4),
        "n_observations": int(model.nobs),
        "means_2x2": {
            "control_pre": round(ctrl_pre, 3) if not np.isnan(ctrl_pre) else None,
            "control_post": round(ctrl_post, 3) if not np.isnan(ctrl_post) else None,
            "control_delta": round(delta_ctrl, 3) if not np.isnan(delta_ctrl) else None,
            "treated_pre": round(treat_pre, 3) if not np.isnan(treat_pre) else None,
            "treated_post": round(treat_post, 3) if not np.isnan(treat_post) else None,
            "treated_delta": round(delta_treat, 3) if not np.isnan(delta_treat) else None,
            "did_empirical": round(empirical_did, 3) if not np.isnan(empirical_did) else None,
        },
        "regression_summary": model.summary().as_text(),
        "trajectories": trajectories,
    }


def validate_parallel_trends(
    df: pd.DataFrame,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    cutoff_time: Any,
) -> Dict[str, Any]:
    """
    Tests the fundamental Parallel Trends assumption in the pre-treatment period.
    Fits a differential slope model: Outcome = a + b * Time + c * Treat + d * (Time * Treat)
    If coefficient 'd' (Time * Treat) has p-value >= 0.05, parallel trends cannot be rejected.
    """
    pre_data = df[df[time_col] < cutoff_time].copy()
    
    if len(pre_data) < 20:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "Not enough pre-treatment periods to robustly test parallel trends.",
            "p_value": None,
        }

    # Normalize time column to numeric 0, 1, 2, ...
    unique_times = sorted(pre_data[time_col].unique())
    time_mapping = {t: i for i, t in enumerate(unique_times)}
    pre_data["time_idx"] = pre_data[time_col].map(time_mapping)
    pre_data["treat_binary"] = pre_data[treatment_col].astype(int)

    formula = f"{outcome_col} ~ time_idx + treat_binary + time_idx:treat_binary"
    model = smf.ols(formula, data=pre_data).fit(cov_type="HC3")

    p_interaction = float(model.pvalues.get("time_idx:treat_binary", 1.0))
    coef_interaction = float(model.params.get("time_idx:treat_binary", 0.0))

    if p_interaction >= 0.05:
        verdict = "PASS"
        recommendation = (
            f"✅ PARALLEL TRENDS VALIDATED (pre-trend differential slope = {coef_interaction:.4f}, p = {p_interaction:.4f} >= 0.05). "
            "Pre-treatment growth rates between treated and control units are statistically indistinguishable. "
            "The DiD counterfactual assumption is sound."
        )
    else:
        verdict = "FAIL"
        recommendation = (
            f"⚠️ PARALLEL TRENDS VIOLATION WARNING (pre-trend differential slope = {coef_interaction:.4f}, p = {p_interaction:.4f} < 0.05). "
            "Treated and control units had significantly different trajectories before feature launch. "
            "Standard DiD may be biased; consider using Synthetic Control instead."
        )

    return {
        "status": verdict,
        "differential_slope": round(coef_interaction, 4),
        "p_value": p_interaction,
        "recommendation": recommendation,
    }


def fit_event_study(
    df: pd.DataFrame,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    cutoff_time: Any,
    reference_period_offset: int = -1,
) -> pd.DataFrame:
    """
    Fits a Dynamic Event Study model with leads and lags around the rollout date:
        Outcome = alpha + sum_{k != -1} beta_k * (Treat * 1(t - t_event == k)) + time_FE + unit_FE

    Returns:
        DataFrame of event study coefficients (lead/lag, estimate, std_error, ci_lower, ci_upper, p_value).
    """
    data = df.copy()
    unique_times = sorted(data[time_col].unique())
    if cutoff_time not in unique_times:
        # Find closest time
        event_idx = [i for i, t in enumerate(unique_times) if t >= cutoff_time][0]
    else:
        event_idx = unique_times.index(cutoff_time)

    # Relative period k
    time_to_rel = {t: i - event_idx for i, t in enumerate(unique_times)}
    data["rel_period"] = data[time_col].map(time_to_rel)
    data["treat_binary"] = data[treatment_col].astype(int)

    # Exclude reference period (typically k = -1)
    rel_periods = sorted([k for k in data["rel_period"].unique() if k != reference_period_offset])

    # Create dummy interaction terms
    col_names = []
    for k in rel_periods:
        prefix = f"lead_{abs(k)}" if k < 0 else f"lag_{k}"
        col_name = f"rel_{prefix}"
        data[col_name] = (data["treat_binary"] == 1) & (data["rel_period"] == k)
        data[col_name] = data[col_name].astype(int)
        col_names.append((k, col_name))

    # OLS regression with time and relative period interaction dummies
    formula = f"{outcome_col} ~ treat_binary + C(rel_period) + " + " + ".join([c[1] for c in col_names])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=data).fit(cov_type="HC3")

    results = []
    for k, col_name in col_names:
        coef = float(model.params.get(col_name, 0.0))
        se = float(model.bse.get(col_name, 0.0))
        pval = float(model.pvalues.get(col_name, 1.0))
        ci = model.conf_int().loc[col_name].values if col_name in model.conf_int().index else [0.0, 0.0]

        results.append({
            "relative_period": k,
            "period_label": f"t{k:+d}",
            "is_pre_period": bool(k < 0),
            "estimate": round(coef, 4),
            "std_error": round(se, 4),
            "ci_lower": round(float(ci[0]), 4),
            "ci_upper": round(float(ci[1]), 4),
            "p_value": float(pval),
        })

    # Add reference period (k = -1, estimate = 0)
    results.append({
        "relative_period": reference_period_offset,
        "period_label": f"t{reference_period_offset:+d} (Ref)",
        "is_pre_period": True,
        "estimate": 0.0,
        "std_error": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "p_value": 1.0,
    })

    results_df = pd.DataFrame(results).sort_values("relative_period").reset_index(drop=True)
    return results_df
