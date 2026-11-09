"""
Core Causal Inference & Experimentation Platform Engine
Module: Sequential Testing (mSPRT & Alpha Spending) & Power Analysis
Reference: Johari, Pekelis, Walsh (2017); O'Brien & Fleming (1979)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_sequential_bounds(
    time_points: List[Any],
    z_scores: List[float],
    total_planned_periods: int,
    alpha: float = 0.05,
    method: str = "obrien_fleming",
) -> pd.DataFrame:
    """
    Computes safe continuous monitoring boundaries to prevent Type I error inflation
    caused by daily peeking at experiment metrics.

    Methods:
        - 'obrien_fleming': Conservative at early stages, converges to nominal z-critical near completion.
        - 'pocock': Constant boundary across all interim evaluation checkpoints.

    Args:
        time_points: List of timestamps / dates / checkpoints
        z_scores: Observed cumulative z-score at each checkpoint
        total_planned_periods: Total expected duration / checkpoints (T)
        alpha: Overall experiment significance level (default 0.05)
        method: 'obrien_fleming' or 'pocock'

    Returns:
        DataFrame with upper/lower stopping bounds, observed z-scores, and early stopping decisions.
    """
    k = len(time_points)
    total_planned = max(total_planned_periods, k)

    records = []
    stopped = False
    stop_period = None
    stop_reason = None

    for i, t in enumerate(time_points):
        info_fraction = min(1.0, (i + 1) / float(total_planned))
        info_fraction = max(0.01, info_fraction)

        # Compute critical boundaries
        if method == "obrien_fleming":
            # Lan-DeMets approximation to O'Brien-Fleming
            # alpha(t) = 2 - 2 * Phi(z_{alpha/2} / sqrt(t))
            z_nominal = stats.norm.ppf(1.0 - alpha / 2.0)
            boundary = z_nominal / np.sqrt(info_fraction)
        else:  # Pocock
            # Pocock boundary maintains roughly constant critical value
            # Adjusted for multiple looks
            num_looks = total_planned
            pocock_inflation = 1.0 + 0.3 * np.log(num_looks)
            boundary = stats.norm.ppf(1.0 - (alpha / pocock_inflation) / 2.0)

        boundary = float(boundary)
        z_obs = float(z_scores[i])

        can_stop_early = abs(z_obs) >= boundary

        if can_stop_early and not stopped:
            stopped = True
            stop_period = str(t)
            stop_reason = "Efficacy Lift Detected" if z_obs > 0 else "Safety / Harm Stop"

        records.append({
            "checkpoint": str(t),
            "period_index": i + 1,
            "information_fraction": round(info_fraction, 3),
            "observed_z_score": round(z_obs, 3),
            "upper_bound": round(boundary, 3),
            "lower_bound": round(-boundary, 3),
            "can_stop_early": bool(can_stop_early),
            "stopped": bool(stopped),
        })

    return pd.DataFrame(records)


def compute_power_and_sample_size(
    baseline_mean: float,
    baseline_std: float,
    mde_relative_pct: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_traffic: int = 5000,
    cuped_rho: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes required sample size, experiment duration, and CUPED acceleration gains.

    Formula:
        N = 2 * (z_{1 - alpha/2} + z_{power})^2 * sigma^2 / delta^2
        N_cuped = N * (1 - rho^2)

    Args:
        baseline_mean: Expected control group metric average
        baseline_std: Metric standard deviation (sigma)
        mde_relative_pct: Minimum Detectable Effect as a % of baseline mean (e.g. 2.0 for +2%)
        alpha: False positive rate (Type I error, default 0.05)
        power: Statistical power 1 - beta (Type II error, default 0.80)
        daily_traffic: Estimated total active users entering the experiment per day
        cuped_rho: Expected pre/post correlation rho (0.0 if not using CUPED)

    Returns:
        Dict with sample sizes, runtime in days, and velocity comparison.
    """
    delta_abs = abs(baseline_mean * (mde_relative_pct / 100.0))
    if delta_abs <= 1e-9:
        raise ValueError("MDE cannot be zero.")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    # Required sample size per variant
    n_per_variant = 2.0 * ((z_alpha + z_beta) ** 2) * (baseline_std ** 2) / (delta_abs ** 2)
    n_total_raw = int(np.ceil(2.0 * n_per_variant))

    # Apply CUPED compression: N_cuped = N_raw * (1 - rho^2)
    rho_clipped = np.clip(cuped_rho, 0.0, 0.99)
    variance_reduction = float(rho_clipped ** 2)
    n_total_cuped = int(np.ceil(n_total_raw * (1.0 - variance_reduction)))

    # Runtime in days
    days_raw = int(np.ceil(n_total_raw / float(daily_traffic))) if daily_traffic > 0 else 0
    days_cuped = int(np.ceil(n_total_cuped / float(daily_traffic))) if daily_traffic > 0 else 0
    days_saved = max(0, days_raw - days_cuped)

    return {
        "mde_relative_pct": mde_relative_pct,
        "mde_absolute": round(delta_abs, 4),
        "alpha": alpha,
        "power": power,
        "raw_experiment": {
            "sample_size_per_variant": int(np.ceil(n_per_variant)),
            "total_sample_size": n_total_raw,
            "estimated_duration_days": days_raw,
        },
        "cuped_experiment": {
            "expected_rho": round(float(rho_clipped), 3),
            "variance_reduction_pct": round(variance_reduction * 100.0, 1),
            "sample_size_per_variant": int(np.ceil(n_per_variant * (1.0 - variance_reduction))),
            "total_sample_size": n_total_cuped,
            "estimated_duration_days": days_cuped,
            "days_saved": days_saved,
            "velocity_multiplier": round(1.0 / (1.0 - variance_reduction), 2),
        },
    }

