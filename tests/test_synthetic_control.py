"""
Unit tests for Synthetic Control Module
"""

import pytest
import pandas as pd
import numpy as np
from core.synthetic_control import fit_synthetic_control
from core.datasets import generate_regional_rollout_data


def test_synthetic_control_fit():
    df = generate_regional_rollout_data(n_regions=6, n_weeks=30, treat_week=20, seed=777)
    res = fit_synthetic_control(
        df=df,
        unit_col="region",
        time_col="week",
        outcome_col="engagement_score",
        treated_unit="US-West",
        treatment_cutoff_time=20,
    )

    # Weights must sum to 1.0 and be non-negative
    weights = res["weights_array"]
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    assert np.all(weights >= -1e-6)

    # Pre-treatment fit must be strong
    assert res["fit_metrics"]["r_squared"] > 0.85
    assert res["fit_metrics"]["pre_rmse"] < 5.0

    # Post-treatment lift should be positive and detected
    assert res["post_metrics"]["average_treatment_effect"] > 8.0

