"""
Unit tests for CUPED Variance Reduction Module
"""

import pytest
import pandas as pd
import numpy as np
from core.cuped import apply_cuped, compute_sample_compression_savings
from core.datasets import generate_search_ranking_experiment


def test_cuped_variance_reduction():
    df = generate_search_ranking_experiment(n_users=5000, seed=42)
    res = apply_cuped(
        df=df,
        treatment_col="is_treatment",
        outcome_col="post_experiment_queries",
        covariate_col="pre_experiment_queries",
        treatment_val=1,
        control_val=0,
    )

    # Pre/post correlation should be very high
    assert res["correlation_rho"] > 0.70
    assert res["variance_reduction_pct"] > 50.0
    assert res["sample_size_compression"] > 2.0

    # CUPED confidence interval should be strictly narrower than raw CI
    assert res["cuped"]["ci_width"] < res["raw"]["ci_width"]
    assert res["cuped"]["ci_reduction_pct"] > 30.0


def test_multi_covariate_cuped():
    df = generate_search_ranking_experiment(n_users=5000, seed=42)
    res = apply_cuped(
        df=df,
        treatment_col="is_treatment",
        outcome_col="post_experiment_queries",
        covariate_col=["pre_experiment_queries", "latency_ms"],
        treatment_val=1,
        control_val=0,
    )
    assert res["is_multi_covariate"]
    assert len(res["covariates_used"]) == 2
    assert res["variance_reduction_pct"] > 50.0
    assert res["cuped"]["ci_width"] < res["raw"]["ci_width"]


def test_sample_compression_savings():
    savings = compute_sample_compression_savings(
        current_sample_size=100000,
        current_duration_days=14,
        variance_reduction_pct=75.0,
    )
    assert savings["compressed_sample_size"] == 25000
    assert savings["users_saved"] == 75000
    assert savings["compressed_duration_days"] < 14
    assert savings["days_saved"] > 0
    assert savings["velocity_multiplier"] == 4.0

