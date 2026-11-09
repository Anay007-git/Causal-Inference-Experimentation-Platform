"""
Core Causal Inference & Experimentation Platform Engine
Module: SRM (Sample Ratio Mismatch)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def calculate_srm(
    observed: Dict[str, int],
    expected_ratios: Optional[Dict[str, float]] = None,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """
    Computes Pearson's Chi-square Goodness-of-Fit test and Binomial test for SRM.

    Args:
        observed: Dict mapping variant name to observed user count (e.g., {'control': 49500, 'treatment': 50500})
        expected_ratios: Dict mapping variant name to expected traffic ratio (e.g., {'control': 0.5, 'treatment': 0.5}).
                         If None, equal allocation is assumed.
        alpha: Critical threshold (default 0.01 following industry standard to avoid false positives).

    Returns:
        Dict with test statistic, p-value, decision status, and per-variant details.
    """
    variants = list(observed.keys())
    if len(variants) < 2:
        raise ValueError("At least two variants are required for SRM testing.")

    obs_counts = np.array([observed[v] for v in variants], dtype=float)
    total_obs = np.sum(obs_counts)

    if total_obs == 0:
        raise ValueError("Total observed count cannot be zero.")

    if expected_ratios is None:
        exp_props = np.ones(len(variants)) / len(variants)
    else:
        exp_props = np.array([expected_ratios.get(v, 1.0 / len(variants)) for v in variants], dtype=float)
        exp_props = exp_props / np.sum(exp_props)

    exp_counts = total_obs * exp_props

    # Pearson's Chi-square Goodness-of-Fit test
    chi2_stat, p_value = stats.chisquare(f_obs=obs_counts, f_exp=exp_counts)

    # If exactly 2 variants, also compute exact binomial test
    binomial_pval = None
    if len(variants) == 2:
        binom_res = stats.binomtest(int(obs_counts[1]), n=int(total_obs), p=exp_props[1])
        binomial_pval = float(binom_res.pvalue)

    # Decision classification
    if p_value < 0.001:
        status = "CRITICAL_SRM"
        severity = "Critical"
        recommendation = (
            "🚨 CRITICAL SAMPLE RATIO MISMATCH DETECTED (p < 0.001). "
            "Traffic allocation is severely compromised. Do NOT evaluate primary metrics; "
            "average treatment effect (ATE) is invalid due to selection bias or telemetry loss."
        )
    elif p_value < alpha:
        status = "WARNING_SRM"
        severity = "Warning"
        recommendation = (
            f"⚠️ SAMPLE RATIO MISMATCH WARNING (p = {p_value:.4f} < {alpha}). "
            "Allocation deviates significantly from design. Inspect traffic telemetry before making decisions."
        )
    else:
        status = "PASS"
        severity = "Healthy"
        recommendation = (
            f"✅ SRM CHECK PASSED (p = {p_value:.4f} >= {alpha}). "
            "Observed traffic matches designed allocation proportions within expected statistical variance."
        )

    # Per-variant breakdown
    variant_details = []
    for i, v in enumerate(variants):
        obs = obs_counts[i]
        exp = exp_counts[i]
        diff = obs - exp
        pct_diff = (diff / exp) * 100.0
        obs_ratio = obs / total_obs
        variant_details.append({
            "variant": v,
            "observed_count": int(obs),
            "expected_count": round(exp, 1),
            "observed_ratio": round(obs_ratio, 4),
            "expected_ratio": round(exp_props[i], 4),
            "delta_count": int(diff),
            "delta_pct": round(pct_diff, 2),
            "standardized_residual": round(diff / np.sqrt(exp), 3),
        })

    return {
        "status": status,
        "severity": severity,
        "chi2_stat": round(float(chi2_stat), 4),
        "degrees_of_freedom": len(variants) - 1,
        "p_value": float(p_value),
        "binomial_p_value": binomial_pval,
        "total_sample_size": int(total_obs),
        "alpha": alpha,
        "recommendation": recommendation,
        "variant_details": variant_details,
    }


def calculate_cumulative_srm(
    df: pd.DataFrame,
    date_col: str,
    variant_col: str,
    expected_ratios: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Computes cumulative traffic and daily SRM p-values over time.

    Args:
        df: DataFrame containing daily experiment logs
        date_col: Column name representing date or timestamp
        variant_col: Column name representing variant label
        expected_ratios: Expected allocation proportions
    """
    # Group by date and variant
    daily = (
        df.groupby([date_col, variant_col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    variants = list(daily.columns)
    if expected_ratios is None:
        exp_props = np.ones(len(variants)) / len(variants)
    else:
        exp_props = np.array([expected_ratios.get(v, 1.0 / len(variants)) for v in variants], dtype=float)
        exp_props = exp_props / np.sum(exp_props)

    records = []
    cum_counts = np.zeros(len(variants))

    for date, row in daily.iterrows():
        cum_counts += row.values
        total = np.sum(cum_counts)
        exp_counts = total * exp_props

        if total > 0:
            chi2, p_val = stats.chisquare(f_obs=cum_counts, f_exp=exp_counts)
        else:
            chi2, p_val = 0.0, 1.0

        rec = {
            "date": str(date),
            "cumulative_total": int(total),
            "chi2_stat": round(float(chi2), 3),
            "p_value": float(p_val),
            "log10_p_value": float(-np.log10(max(p_val, 1e-15))),
        }
        for i, v in enumerate(variants):
            rec[f"count_{v}"] = int(cum_counts[i])
            rec[f"ratio_{v}"] = round(cum_counts[i] / total, 4) if total > 0 else 0
        records.append(rec)

    return pd.DataFrame(records)


def diagnose_srm_slices(
    df: pd.DataFrame,
    variant_col: str,
    slice_cols: List[str],
    expected_ratios: Optional[Dict[str, float]] = None,
    alpha: float = 0.001,
) -> pd.DataFrame:
    """
    Performs dimensional slice diagnostics to isolate which dimension (e.g. browser, device, country)
    is driving a Sample Ratio Mismatch.

    Returns:
        DataFrame sorted by most severe mismatch (lowest p-value).
    """
    slice_results = []
    variants = sorted(df[variant_col].unique())

    if expected_ratios is None:
        exp_props = np.ones(len(variants)) / len(variants)
    else:
        exp_props = np.array([expected_ratios.get(v, 1.0 / len(variants)) for v in variants], dtype=float)
        exp_props = exp_props / np.sum(exp_props)

    for col in slice_cols:
        if col not in df.columns:
            continue
        grouped = df.groupby([col, variant_col]).size().unstack(fill_value=0)
        for cat_val, row in grouped.iterrows():
            obs = np.array([row.get(v, 0) for v in variants], dtype=float)
            tot = np.sum(obs)
            if tot < 20:  # Skip tiny sample slices
                continue
            exp = tot * exp_props
            chi2, p_val = stats.chisquare(f_obs=obs, f_exp=exp)
            
            slice_results.append({
                "dimension": col,
                "dimension_value": str(cat_val),
                "total_count": int(tot),
                "chi2_stat": round(float(chi2), 3),
                "p_value": float(p_val),
                "has_srm": bool(p_val < alpha),
                "observed_distribution": {str(v): int(obs[i]) for i, v in enumerate(variants)},
                "observed_ratios": {str(v): round(float(obs[i] / tot), 4) for i, v in enumerate(variants)},
            })

    result_df = pd.DataFrame(slice_results)
    if not result_df.empty:
        result_df = result_df.sort_values(by="p_value", ascending=True).reset_index(drop=True)
    return result_df

