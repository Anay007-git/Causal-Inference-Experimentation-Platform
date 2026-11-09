"""
Core Causal Inference & Experimentation Platform Engine
"""

from core.srm import calculate_srm, calculate_cumulative_srm, diagnose_srm_slices
from core.cuped import apply_cuped, compute_sample_compression_savings
from core.did import estimate_did, validate_parallel_trends, fit_event_study
from core.synthetic_control import fit_synthetic_control, run_in_space_placebo_test
from core.causal_graph import run_dowhy_causal_analysis
from core.sequential import compute_sequential_bounds, compute_power_and_sample_size
from core.column_detector import (
    detect_variant_column,
    detect_outcome_column,
    detect_baseline_covariate_columns,
    detect_unit_column,
    detect_time_column,
    detect_slice_columns,
    aggregate_user_data_to_panel,
)
from core.datasets import (
    generate_search_ranking_experiment,
    generate_srm_bug_dataset,
    generate_regional_rollout_data,
    generate_youtube_causal_data,
)

__all__ = [
    "calculate_srm",
    "calculate_cumulative_srm",
    "diagnose_srm_slices",
    "apply_cuped",
    "compute_sample_compression_savings",
    "estimate_did",
    "validate_parallel_trends",
    "fit_event_study",
    "fit_synthetic_control",
    "run_in_space_placebo_test",
    "run_dowhy_causal_analysis",
    "compute_sequential_bounds",
    "compute_power_and_sample_size",
    "detect_variant_column",
    "detect_outcome_column",
    "detect_baseline_covariate_columns",
    "detect_unit_column",
    "detect_time_column",
    "detect_slice_columns",
    "aggregate_user_data_to_panel",
    "generate_search_ranking_experiment",
    "generate_srm_bug_dataset",
    "generate_regional_rollout_data",
    "generate_youtube_causal_data",
]
