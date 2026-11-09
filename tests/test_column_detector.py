"""
Unit Tests for Intelligent Schema Detection & Column Mapping
"""

import pandas as pd
import numpy as np
import pytest
from core.column_detector import (
    detect_variant_column,
    detect_outcome_column,
    detect_baseline_covariate_columns,
    detect_unit_column,
    detect_time_column,
    detect_slice_columns,
    aggregate_user_data_to_panel,
)


def test_detect_variant_column():
    # Canonical naming
    df1 = pd.DataFrame({"variant": ["control", "treatment"], "y": [1.0, 2.0]})
    assert detect_variant_column(df1) == "variant"

    # Boolean naming
    df2 = pd.DataFrame({"is_treatment": [0, 1], "metric": [10, 20]})
    assert detect_variant_column(df2) == "is_treatment"

    # Arbitrary naming with A/B values
    df3 = pd.DataFrame({"arm": ["A", "B", "A"], "val": [1, 2, 3]})
    assert detect_variant_column(df3) == "arm"


def test_detect_outcome_column():
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "post_experiment_revenue": [100.5, 200.0, 150.2],
        "pre_experiment_revenue": [80.0, 190.0, 140.0],
    })
    outcome = detect_outcome_column(df)
    assert outcome == "post_experiment_revenue"


def test_detect_baseline_covariate_columns():
    df = pd.DataFrame({
        "post_queries": [10, 20, 30],
        "pre_queries": [9, 18, 28],
        "other_num": [1, 1, 1],
    })
    covs = detect_baseline_covariate_columns(df, outcome_col="post_queries")
    assert "pre_queries" in covs


def test_detect_unit_and_time_columns():
    df = pd.DataFrame({
        "region": ["US-West", "EU-Central", "APAC"],
        "week": ["2026-W01", "2026-W01", "2026-W01"],
        "engagement": [4.5, 3.2, 5.1],
    })
    assert detect_unit_column(df) == "region"
    assert detect_time_column(df) == "week"


def test_aggregate_user_data_to_panel():
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "country": ["US", "US", "DE", "DE"],
        "date": ["2026-10-01", "2026-10-01", "2026-10-01", "2026-10-01"],
        "spend": [10.0, 20.0, 50.0, 70.0],
        "is_treatment": [1, 1, 0, 0],
    })
    panel = aggregate_user_data_to_panel(
        df,
        unit_col="country",
        time_col="date",
        outcome_col="spend",
        treatment_col="is_treatment",
    )
    assert len(panel) == 2
    assert "avg_spend" in panel.columns
    assert panel.loc[panel["country"] == "US", "avg_spend"].iloc[0] == 15.0

