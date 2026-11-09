"""
Core Causal Inference & Experimentation Platform Engine
Module: Synthetic Control Method (SCM) & In-Space Placebo Permutations
Reference: Abadie, Diamond, Hainmueller (JASA 2010, AJPS 2015)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_synthetic_control(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: str,
    treatment_cutoff_time: Any,
    donor_pool: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fits the Synthetic Control Method via constrained quadratic optimization.

    Minimizes:
        || Y_1,pre - Y_0,pre * W ||_2^2
    Subject to:
        w_j >= 0  and  sum(w_j) = 1

    Args:
        df: Panel DataFrame containing unit, time, and outcome columns
        unit_col: Column name identifying region/market
        time_col: Column name identifying time/period
        outcome_col: Metric of interest
        treated_unit: Identifier of the treated region
        treatment_cutoff_time: Value in time_col when treatment started
        donor_pool: Optional explicit list of donor unit identifiers (otherwise all other units)

    Returns:
        Dict with optimal weights, fit metrics, trajectories, and gap series.
    """
    # Pivot to wide format: index=time, columns=units, values=outcome
    pivot_df = df.pivot(index=time_col, columns=unit_col, values=outcome_col).sort_index()

    if treated_unit not in pivot_df.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not found in data.")

    # Identify donor pool
    if donor_pool is None:
        donors = [u for u in pivot_df.columns if u != treated_unit]
    else:
        donors = [u for u in donor_pool if u in pivot_df.columns and u != treated_unit]

    if len(donors) < 2:
        raise ValueError("At least 2 donor units are required for synthetic control.")

    # Split pre and post periods
    pre_mask = pivot_df.index < treatment_cutoff_time
    post_mask = pivot_df.index >= treatment_cutoff_time

    if not np.any(pre_mask):
        raise ValueError("No pre-treatment periods found before cutoff time.")
    if not np.any(post_mask):
        raise ValueError("No post-treatment periods found after cutoff time.")

    # Pre-treatment matrices
    Y1_pre = pivot_df.loc[pre_mask, treated_unit].values.astype(float)
    Y0_pre = pivot_df.loc[pre_mask, donors].values.astype(float)

    # Full period matrices
    Y1_full = pivot_df[treated_unit].values.astype(float)
    Y0_full = pivot_df[donors].values.astype(float)
    all_times = list(pivot_df.index)

    n_donors = len(donors)

    # Quadratic objective: MSE in pre-treatment period
    def objective(w: np.ndarray) -> float:
        diff = Y1_pre - Y0_pre @ w
        return float(np.mean(diff ** 2))

    # Constraints: sum(w) = 1 and 0 <= w_j <= 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    w_initial = np.ones(n_donors) / n_donors

    # Optimization
    opt_res = minimize(
        objective,
        w_initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    opt_weights = np.maximum(opt_res.x, 0.0)
    opt_weights = opt_weights / np.sum(opt_weights)  # Ensure exact sum to 1

    # Synthetic counterfactual trajectory
    synthetic_trajectory = Y0_full @ opt_weights
    gap_series = Y1_full - synthetic_trajectory

    # Pre-treatment fit diagnostics
    synth_pre = synthetic_trajectory[pre_mask]
    actual_pre = Y1_pre
    pre_mse = float(np.mean((actual_pre - synth_pre) ** 2))
    pre_rmse = float(np.sqrt(pre_mse))
    pre_mae = float(np.mean(np.abs(actual_pre - synth_pre)))
    var_actual_pre = float(np.var(actual_pre))
    r_squared = float(1.0 - (pre_mse / (var_actual_pre + 1e-12)))

    # Post-treatment effect diagnostics
    synth_post = synthetic_trajectory[post_mask]
    actual_post = Y1_full[post_mask]
    post_gap = gap_series[post_mask]
    post_mse = float(np.mean(post_gap ** 2))
    avg_post_effect = float(np.mean(post_gap))
    avg_post_actual = float(np.mean(actual_post))
    pct_lift = float((avg_post_effect / (avg_post_actual - avg_post_effect + 1e-12)) * 100.0)
    cumulative_effect = float(np.sum(post_gap))

    # Weight distribution table
    donor_weights = []
    for d, w in zip(donors, opt_weights):
        if w >= 0.005:  # Only report meaningful weights >= 0.5%
            donor_weights.append({
                "donor_unit": str(d),
                "weight": round(float(w), 4),
                "weight_pct": round(float(w * 100.0), 2),
            })
    donor_weights.sort(key=lambda x: x["weight"], reverse=True)

    # Trajectory DataFrame
    traj_df = pd.DataFrame({
        "time": all_times,
        "is_post": post_mask,
        "treated_actual": Y1_full,
        "synthetic_counterfactual": synthetic_trajectory,
        "causal_gap": gap_series,
    })

    return {
        "treated_unit": treated_unit,
        "treatment_cutoff_time": treatment_cutoff_time,
        "n_pre_periods": int(np.sum(pre_mask)),
        "n_post_periods": int(np.sum(post_mask)),
        "donor_weights": donor_weights,
        "all_donors": donors,
        "weights_array": opt_weights,
        "fit_metrics": {
            "pre_rmse": round(pre_rmse, 4),
            "pre_mae": round(pre_mae, 4),
            "pre_mse": round(pre_mse, 6),
            "r_squared": round(r_squared, 4),
        },
        "post_metrics": {
            "average_treatment_effect": round(avg_post_effect, 4),
            "cumulative_lift": round(cumulative_effect, 4),
            "percentage_lift": round(pct_lift, 2),
            "post_mse": round(post_mse, 6),
            "post_to_pre_mse_ratio": round(post_mse / (pre_mse + 1e-12), 3),
        },
        "trajectories": traj_df,
    }


def run_in_space_placebo_test(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: str,
    treatment_cutoff_time: Any,
    max_donors: int = 25,
) -> Dict[str, Any]:
    """
    Conducts Abadie In-Space Placebo Permutation Tests:
    Fits Synthetic Control iteratively for every donor unit as if it received treatment.
    Calculates the exact permutation p-value based on post/pre MSE ratio.

    Returns:
        Dict with permutation p-value, rank, and all placebo gap series for visualization.
    """
    # 1. Fit synthetic control for the true treated unit
    main_res = fit_synthetic_control(
        df=df,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        treatment_cutoff_time=treatment_cutoff_time,
    )

    true_pre_mse = main_res["fit_metrics"]["pre_mse"]
    true_post_mse = main_res["post_metrics"]["post_mse"]
    true_ratio = true_post_mse / (true_pre_mse + 1e-12)

    donors = main_res["all_donors"][:max_donors]
    times = list(main_res["trajectories"]["time"])
    post_mask = main_res["trajectories"]["is_post"].values

    placebo_results = []
    ratios = [true_ratio]

    # Structure to hold gap curves for plotting
    placebo_cols = {
        "time": times,
        treated_unit: main_res["trajectories"]["causal_gap"].values,
    }

    for d in donors:
        try:
            p_res = fit_synthetic_control(
                df=df,
                unit_col=unit_col,
                time_col=time_col,
                outcome_col=outcome_col,
                treated_unit=d,
                treatment_cutoff_time=treatment_cutoff_time,
            )
            p_pre_mse = p_res["fit_metrics"]["pre_mse"]
            p_post_mse = p_res["post_metrics"]["post_mse"]
            p_ratio = p_post_mse / (p_pre_mse + 1e-12)

            ratios.append(p_ratio)
            placebo_cols[f"Placebo_{d}"] = p_res["trajectories"]["causal_gap"].values

            placebo_results.append({
                "unit": d,
                "pre_mse": p_pre_mse,
                "post_mse": p_post_mse,
                "mse_ratio": p_ratio,
            })
        except Exception:
            continue

    placebo_gaps_df = pd.DataFrame(placebo_cols)

    total_units = len(ratios)
    # Permutation p-value: fraction of units with post/pre MSE ratio >= true treated ratio
    rank = sum(1 for r in ratios if r >= true_ratio)
    permutation_p_val = float(rank / total_units)

    if permutation_p_val <= 0.05:
        verdict = "STATISTICALLY_SIGNIFICANT"
        recommendation = (
            f"✅ CAUSAL EFFECT CONFIRMED VIA PLACEBO PERMUTATION (p = {permutation_p_val:.4f} <= 0.05). "
            f"The post-launch divergence for '{treated_unit}' ranks #{rank} out of {total_units} units. "
            "It is highly improbable that random regional noise generated this lift."
        )
    else:
        verdict = "INCONCLUSIVE"
        recommendation = (
            f"⚠️ PLACEBO PERMUTATION INCONCLUSIVE (p = {permutation_p_val:.4f} > 0.05). "
            f"The post-launch divergence for '{treated_unit}' ranks #{rank} of {total_units}. "
            "Donor units experienced comparable natural fluctuations in the post-period."
        )

    return {
        "status": verdict,
        "permutation_p_value": round(permutation_p_val, 4),
        "rank": rank,
        "total_units_tested": total_units,
        "treated_ratio": round(true_ratio, 3),
        "recommendation": recommendation,
        "placebo_gaps": placebo_gaps_df,
        "main_result": main_res,
    }

