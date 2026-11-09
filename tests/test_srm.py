"""
Unit tests for Sample Ratio Mismatch (SRM) Module
"""

import pytest
import pandas as pd
import numpy as np
from core.srm import calculate_srm, calculate_cumulative_srm, diagnose_srm_slices
from core.datasets import generate_srm_bug_dataset


def test_srm_balanced():
    # 50,000 vs 50,050 users (healthy random variance)
    res = calculate_srm({"control": 50000, "treatment": 50050}, alpha=0.01)
    assert res["status"] == "PASS"
    assert res["p_value"] > 0.05
    assert len(res["variant_details"]) == 2


def test_srm_severe_mismatch():
    # 48,000 vs 52,000 (severe allocation mismatch)
    res = calculate_srm({"control": 48000, "treatment": 52000}, alpha=0.01)
    assert res["status"] == "CRITICAL_SRM"
    assert res["p_value"] < 1e-10
    assert res["chi2_stat"] > 50.0


def test_srm_custom_ratios():
    # 90/10 split design
    observed = {"control": 90050, "treatment": 9950}
    ratios = {"control": 0.9, "treatment": 0.1}
    res = calculate_srm(observed, expected_ratios=ratios, alpha=0.01)
    assert res["status"] == "PASS"
    assert res["p_value"] > 0.05


def test_srm_slice_diagnostics():
    df = generate_srm_bug_dataset(n_users=15000, seed=42)
    diag_df = diagnose_srm_slices(df, variant_col="variant", slice_cols=["device_type", "browser"])
    assert not diag_df.empty
    assert "has_srm" in diag_df.columns
    # Safari should trip SRM due to the injected redirect bug
    safari_slices = diag_df[diag_df["dimension_value"] == "Safari"]
    assert not safari_slices.empty

