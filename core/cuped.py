"""
Core Causal Inference & Experimentation Platform Engine
Module: CUPED (Controlled-experiment Using Pre-Experiment Data)
Reference: Deng, Xu, Kohavi, Walker (WSDM 2013)
Supports Single & Multi-Covariate Variance Reduction
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats


def apply_cuped(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_col: Union[str, List[str]],
    treatment_val: Any = 1,
    control_val: Any = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Applies single- or multi-covariate CUPED variance reduction using pre-experiment baseline features.

    Adjusted metric formula:
        Y_cuped = Y - (X - E[X]) @ theta
        theta = Var(X)^(-1) Cov(X, Y)

    Args:
        df: Pandas DataFrame containing treatment, outcome (post), and covariate(s) (pre)
        treatment_col: Name of binary treatment assignment column
        outcome_col: Name of post-experiment outcome variable (Y)
        covariate_col: Single column name or list of column names representing baseline covariates
        treatment_val: Value in treatment_col representing treated users (e.g. 1 or 'treatment')
        control_val: Value in treatment_col representing control users (e.g. 0 or 'control')
        alpha: Significance level for confidence intervals (default 0.05 for 95% CI)

    Returns:
        Dict containing raw vs CUPED metrics, variance reduction %, sample size multiplier,
        confidence intervals, and adjusted Series for plotting.
    """
    if isinstance(covariate_col, str):
        cov_cols = [covariate_col]
    else:
        cov_cols = list(covariate_col)

    if not cov_cols:
        raise ValueError("At least one covariate must be specified.")

    required_cols = [treatment_col, outcome_col] + cov_cols
    clean_df = df[required_cols].dropna().copy()

    Y = clean_df[outcome_col].values.astype(float)
    X = clean_df[cov_cols].values.astype(float)
    T = (clean_df[treatment_col] == treatment_val).values

    n_treat = int(np.sum(T))
    n_ctrl = int(np.sum(~T))
    n_total = len(clean_df)

    if n_treat < 10 or n_ctrl < 10:
        raise ValueError("Insufficient sample size in treatment or control groups.")

    var_y_raw = float(np.var(Y, ddof=1))
    if var_y_raw <= 1e-12:
        raise ValueError("Outcome variance is virtually zero.")

    # Center covariates
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute optimal theta via least-squares / covariance inversion
    if len(cov_cols) == 1:
        x_vec = X_centered.flatten()
        var_x = float(np.var(x_vec, ddof=1))
        if var_x <= 1e-12:
            raise ValueError("Covariate variance is virtually zero. Cannot compute CUPED.")
        cov_yx = float(np.cov(Y, x_vec)[0, 1])
        theta_val = cov_yx / var_x
        theta_vec = np.array([theta_val])
        corr_yx = cov_yx / (np.sqrt(var_y_raw) * np.sqrt(var_x) + 1e-12)
        corr_yx = np.clip(corr_yx, -1.0, 1.0)
        rho_squared = float(corr_yx ** 2)
        theta_display = round(float(theta_val), 5)
    else:
        # Multi-covariate OLS projection
        y_centered = Y - np.mean(Y)
        theta_vec, residuals, rank, s = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        theta_display = [round(float(t), 5) for t in theta_vec]
        y_pred = X_centered @ theta_vec
        rho_squared = float(np.var(y_pred) / (var_y_raw + 1e-12))
        rho_squared = np.clip(rho_squared, 0.0, 0.999)
        corr_yx = np.sqrt(rho_squared)

    # CUPED Adjusted metric: Y_cuped = Y - X_centered @ theta
    Y_cuped = Y - (X_centered @ theta_vec)
    clean_df["Y_cuped"] = Y_cuped

    # Variance of adjusted metric
    var_y_cuped = float(np.var(Y_cuped, ddof=1))
    variance_reduction_pct = float((1.0 - (var_y_cuped / var_y_raw)) * 100.0)
    variance_reduction_pct = max(0.0, variance_reduction_pct)

    # Sample size compression multiplier: 1 / (1 - rho^2)
    if rho_squared < 0.999:
        sample_size_compression = float(1.0 / (1.0 - rho_squared))
        traffic_savings_pct = float(rho_squared * 100.0)
    else:
        sample_size_compression = 100.0
        traffic_savings_pct = 99.0

    # Critical z-score for (1 - alpha) confidence interval
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # --- Baseline Raw A/B Analysis ---
    Y_treat = Y[T]
    Y_ctrl = Y[~T]
    mean_raw_treat = float(np.mean(Y_treat))
    mean_raw_ctrl = float(np.mean(Y_ctrl))
    raw_delta = mean_raw_treat - mean_raw_ctrl
    raw_pct_lift = (raw_delta / mean_raw_ctrl) * 100.0 if mean_raw_ctrl != 0 else 0.0

    var_raw_treat = np.var(Y_treat, ddof=1)
    var_raw_ctrl = np.var(Y_ctrl, ddof=1)
    se_raw = float(np.sqrt(var_raw_treat / n_treat + var_raw_ctrl / n_ctrl))
    raw_ci_lower = raw_delta - z_crit * se_raw
    raw_ci_upper = raw_delta + z_crit * se_raw
    raw_ci_width = raw_ci_upper - raw_ci_lower
    raw_t_stat = raw_delta / (se_raw + 1e-12)
    raw_p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(raw_t_stat))))

    # --- CUPED Adjusted A/B Analysis ---
    Y_cuped_treat = Y_cuped[T]
    Y_cuped_ctrl = Y_cuped[~T]
    mean_cuped_treat = float(np.mean(Y_cuped_treat))
    mean_cuped_ctrl = float(np.mean(Y_cuped_ctrl))
    cuped_delta = mean_cuped_treat - mean_cuped_ctrl
    cuped_pct_lift = (cuped_delta / mean_cuped_ctrl) * 100.0 if mean_cuped_ctrl != 0 else 0.0

    var_cuped_treat = np.var(Y_cuped_treat, ddof=1)
    var_cuped_ctrl = np.var(Y_cuped_ctrl, ddof=1)
    se_cuped = float(np.sqrt(var_cuped_treat / n_treat + var_cuped_ctrl / n_ctrl))
    cuped_ci_lower = cuped_delta - z_crit * se_cuped
    cuped_ci_upper = cuped_delta + z_crit * se_cuped
    cuped_ci_width = cuped_ci_upper - cuped_ci_lower
    cuped_t_stat = cuped_delta / (se_cuped + 1e-12)
    cuped_p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(cuped_t_stat))))

    # CI narrowing percentage
    ci_reduction_pct = float(((raw_ci_width - cuped_ci_width) / raw_ci_width) * 100.0)

    return {
        "theta": theta_display,
        "covariates_used": cov_cols,
        "is_multi_covariate": bool(len(cov_cols) > 1),
        "correlation_rho": round(float(corr_yx), 4),
        "rho_squared": round(float(rho_squared), 4),
        "variance_reduction_pct": round(float(variance_reduction_pct), 2),
        "sample_size_compression": round(float(sample_size_compression), 2),
        "traffic_savings_pct": round(float(traffic_savings_pct), 2),
        "sample_size": {
            "total": n_total,
            "treatment": n_treat,
            "control": n_ctrl,
        },
        "raw": {
            "treatment_mean": round(mean_raw_treat, 4),
            "control_mean": round(mean_raw_ctrl, 4),
            "delta": round(raw_delta, 4),
            "pct_lift": round(raw_pct_lift, 3),
            "std_error": round(se_raw, 5),
            "ci_lower": round(raw_ci_lower, 4),
            "ci_upper": round(raw_ci_upper, 4),
            "ci_width": round(raw_ci_width, 4),
            "p_value": raw_p_value,
            "is_significant": bool(raw_p_value < alpha),
        },
        "cuped": {
            "treatment_mean": round(mean_cuped_treat, 4),
            "control_mean": round(mean_cuped_ctrl, 4),
            "delta": round(cuped_delta, 4),
            "pct_lift": round(cuped_pct_lift, 3),
            "std_error": round(se_cuped, 5),
            "ci_lower": round(cuped_ci_lower, 4),
            "ci_upper": round(cuped_ci_upper, 4),
            "ci_width": round(cuped_ci_width, 4),
            "ci_reduction_pct": round(ci_reduction_pct, 2),
            "p_value": cuped_p_value,
            "is_significant": bool(cuped_p_value < alpha),
        },
        "adjusted_data": clean_df,
    }


def compute_sample_compression_savings(
    current_sample_size: int,
    current_duration_days: int,
    variance_reduction_pct: float,
) -> Dict[str, Any]:
    """
    Computes exact engineering and product velocity gains achieved through CUPED.
    """
    vr_frac = min(variance_reduction_pct / 100.0, 0.95)
    needed_sample_size = int(current_sample_size * (1.0 - vr_frac))
    needed_duration_days = max(1, int(round(current_duration_days * (1.0 - vr_frac))))
    days_saved = current_duration_days - needed_duration_days
    users_saved = current_sample_size - needed_sample_size

    return {
        "original_sample_size": current_sample_size,
        "compressed_sample_size": needed_sample_size,
        "users_saved": users_saved,
        "original_duration_days": current_duration_days,
        "compressed_duration_days": needed_duration_days,
        "days_saved": days_saved,
        "velocity_multiplier": round(1.0 / (1.0 - vr_frac), 2),
    }
