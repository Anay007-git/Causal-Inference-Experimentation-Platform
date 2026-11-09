"""
Unit Tests for DoWhy Causal Graph Identification and Refutation
"""

import pandas as pd
import pytest
from core.causal_graph import run_dowhy_causal_analysis
from core.datasets import generate_youtube_causal_data


def test_dowhy_causal_pipeline():
    df = generate_youtube_causal_data(n_users=500, seed=42)
    res = run_dowhy_causal_analysis(
        df=df,
        treatment="recommendation_v2",
        outcome="watch_time_hours",
        common_causes=["tenure_months", "network_speed_mbps"],
        estimation_method="backdoor.linear_regression",
        run_refutations=True,
    )

    assert "causal_estimate" in res
    assert isinstance(res["causal_estimate"], float)
    assert "graph" in res
    assert "nodes" in res["graph"]
    assert "edges" in res["graph"]
    assert "dot_format" in res["graph"]
    assert len(res["refutations"]) == 3
    assert "verdict" in res

