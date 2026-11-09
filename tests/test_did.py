"""
Unit tests for Difference-in-Differences (DiD) Module
"""

import pytest
import pandas as pd
import numpy as np
from core.did import estimate_did, validate_parallel_trends, fit_event_study
from core.datasets import generate_regional_rollout_data


def test_did_estimation():
    df = generate_regional_rollout_data(n_regions=6, n_weeks=30, treat_week=20, seed=777)
    res = estimate_did(
        df=df,
        unit_col="region",
        time_col="week",
        treatment_col="is_treated_region",
        outcome_col="engagement_score",
        cutoff_time=20,
    )

    # Known ground truth lift is ~14.5
    assert 10.0 < res["did_coefficient"] < 18.0
    assert res["is_significant"]
    assert res["p_value"] < 0.05
    assert not res["trajectories"].empty


def test_parallel_trends_validation():
    df = generate_regional_rollout_data(n_regions=6, n_weeks=30, treat_week=20, seed=777)
    pt = validate_parallel_trends(
        df=df,
        time_col="week",
        treatment_col="is_treated_region",
        outcome_col="engagement_score",
        cutoff_time=20,
    )
    # The generated dataset has valid parallel pre-trends
    assert pt["status"] == "PASS"
    assert pt["p_value"] >= 0.05


def test_event_study():
    df = generate_regional_rollout_data(n_regions=6, n_weeks=30, treat_week=20, seed=777)
    ev_df = fit_event_study(
        df=df,
        time_col="week",
        treatment_col="is_treated_region",
        outcome_col="engagement_score",
        cutoff_time=20,
    )
    assert not ev_df.empty
    assert "estimate" in ev_df.columns
    assert "relative_period" in ev_df.columns
