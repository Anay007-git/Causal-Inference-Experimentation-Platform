"""
Unit tests for Sequential Testing & Power Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
from core.sequential import compute_sequential_bounds, compute_power_and_sample_size


def test_sequential_bounds():
    checkpoints = [f"Day_{i}" for i in range(1, 15)]
    # Z-scores showing steady divergence
    z_scores = [0.2, 0.5, 0.8, 1.2, 1.6, 2.1, 2.7, 3.2, 3.5, 3.8, 4.0, 4.2, 4.3, 4.4]

    res_df = compute_sequential_bounds(
        time_points=checkpoints,
        z_scores=z_scores,
        total_planned_periods=14,
        alpha=0.05,
        method="obrien_fleming",
    )

    assert not res_df.empty
    assert len(res_df) == 14
    assert "upper_bound" in res_df.columns
    assert "observed_z_score" in res_df.columns
    # Check that early stopping was triggered as z reached 3.2
    assert res_df["stopped"].any()


def test_power_and_sample_size_calculator():
    res = compute_power_and_sample_size(
        baseline_mean=100.0,
        baseline_std=20.0,
        mde_relative_pct=2.0,  # 2% lift = 2.0 abs
        alpha=0.05,
        power=0.80,
        daily_traffic=1000,
        cuped_rho=0.80,
    )

    assert res["raw_experiment"]["total_sample_size"] > 0
    assert res["cuped_experiment"]["total_sample_size"] < res["raw_experiment"]["total_sample_size"]
    assert res["cuped_experiment"]["variance_reduction_pct"] == 64.0  # 0.8^2
    assert res["cuped_experiment"]["days_saved"] > 0
    assert res["cuped_experiment"]["velocity_multiplier"] > 2.0

