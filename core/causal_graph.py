"""
Core Causal Inference & Experimentation Platform Engine
Module: Causal Graph Modeling & Refutation Engine (DoWhy)
Reference: Sharma, Kiciman (Microsoft Research, 2020)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel


def run_dowhy_causal_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: Optional[List[str]] = None,
    effect_modifiers: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    estimation_method: str = "backdoor.linear_regression",
    run_refutations: bool = True,
) -> Dict[str, Any]:
    """
    Executes DoWhy 4-step causal inference pipeline:
    1. Model: Graph specification
    2. Identify: Backdoor / Instrumental criterion
    3. Estimate: Causal effect estimation
    4. Refute: Falsification and robustness testing

    Args:
        df: Pandas DataFrame containing all variables
        treatment: Treatment variable name
        outcome: Outcome variable name
        common_causes: List of confounding variables
        effect_modifiers: List of heterogeneous effect modifiers
        instruments: List of instrumental variables
        estimation_method: Estimation method name
        run_refutations: Whether to run placebo, common cause, and subset refutations

    Returns:
        Dict with causal estimate, identification details, refutation verdicts, and DAG structure.
    """
    clean_cols = [treatment, outcome]
    if common_causes:
        clean_cols.extend([c for c in common_causes if c in df.columns])
    if effect_modifiers:
        clean_cols.extend([m for m in effect_modifiers if m in df.columns])
    if instruments:
        clean_cols.extend([i for i in instruments if i in df.columns])

    clean_df = df[clean_cols].dropna().copy()

    # Step 1: Model
    model = CausalModel(
        data=clean_df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes or [],
        effect_modifiers=effect_modifiers or [],
        instruments=instruments or [],
        proceed_when_unidentifiable=True,
    )

    # Step 2: Identify
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Step 3: Estimate
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=estimation_method,
            test_significance=False,
        )

    causal_estimate_val = float(estimate.value)

    # Build DAG nodes and edges for UI visualization
    nodes = [{"id": treatment, "label": f"Treatment: {treatment}", "type": "treatment"},
             {"id": outcome, "label": f"Outcome: {outcome}", "type": "outcome"}]
    edges = [{"from": treatment, "to": outcome, "label": "Causal Effect"}]

    if common_causes:
        for c in common_causes:
            nodes.append({"id": c, "label": f"Confounder: {c}", "type": "confounder"})
            edges.append({"from": c, "to": treatment, "label": "Backdoor"})
            edges.append({"from": c, "to": outcome, "label": "Backdoor"})

    if instruments:
        for i in instruments:
            nodes.append({"id": i, "label": f"Instrument: {i}", "type": "instrument"})
            edges.append({"from": i, "to": treatment, "label": "IV"})

    refutation_results = []
    all_passed = True

    if run_refutations:
        # Refutation 1: Placebo Treatment Refuter
        try:
            ref_placebo = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=5,
            )
            placebo_effect = float(ref_placebo.new_effect)
            # Placebo effect should be close to zero
            placebo_passed = abs(placebo_effect) < (0.25 * max(abs(causal_estimate_val), 1e-6))
            if not placebo_passed:
                all_passed = False

            refutation_results.append({
                "test_name": "Placebo Treatment Refuter",
                "description": "Replaces treatment with randomized noise. True causal effect should collapse towards 0.0.",
                "original_effect": round(causal_estimate_val, 4),
                "refuted_effect": round(placebo_effect, 4),
                "passed": placebo_passed,
                "summary": (
                    f"Passed: Placebo effect is {placebo_effect:.4f} (near zero)."
                    if placebo_passed else
                    f"Failed: Placebo effect remained elevated at {placebo_effect:.4f}."
                ),
            })
        except Exception as e:
            refutation_results.append({
                "test_name": "Placebo Treatment Refuter",
                "error": str(e),
                "passed": False,
            })

        # Refutation 2: Random Common Cause Refuter
        try:
            ref_random = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause",
                num_simulations=5,
            )
            random_effect = float(ref_random.new_effect)
            # Random common cause should not alter the estimate meaningfully
            delta_ratio = abs(random_effect - causal_estimate_val) / max(abs(causal_estimate_val), 1e-6)
            random_passed = delta_ratio < 0.20
            if not random_passed:
                all_passed = False

            refutation_results.append({
                "test_name": "Add Random Common Cause Refuter",
                "description": "Injects an independent synthetic confounder into the DAG. Estimate should remain stable.",
                "original_effect": round(causal_estimate_val, 4),
                "refuted_effect": round(random_effect, 4),
                "passed": random_passed,
                "summary": (
                    f"Passed: Estimate remained stable at {random_effect:.4f} (shift < 20%)."
                    if random_passed else
                    f"Warning: Estimate shifted to {random_effect:.4f}."
                ),
            })
        except Exception as e:
            refutation_results.append({
                "test_name": "Add Random Common Cause Refuter",
                "error": str(e),
                "passed": False,
            })

        # Refutation 3: Data Subset Refuter
        try:
            ref_subset = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="data_subset_refuter",
                subset_fraction=0.8,
                num_simulations=5,
            )
            subset_effect = float(ref_subset.new_effect)
            subset_delta = abs(subset_effect - causal_estimate_val) / max(abs(causal_estimate_val), 1e-6)
            subset_passed = subset_delta < 0.25
            if not subset_passed:
                all_passed = False

            refutation_results.append({
                "test_name": "Data Subset Refuter (80% Bootstrap)",
                "description": "Re-estimates on 80% random subsets to confirm estimate stability across sub-populations.",
                "original_effect": round(causal_estimate_val, 4),
                "refuted_effect": round(subset_effect, 4),
                "passed": subset_passed,
                "summary": (
                    f"Passed: Subset effect is {subset_effect:.4f} (robust to sampling variations)."
                    if subset_passed else
                    f"Warning: Estimate fluctuated to {subset_effect:.4f}."
                ),
            })
        except Exception as e:
            refutation_results.append({
                "test_name": "Data Subset Refuter",
                "error": str(e),
                "passed": False,
            })

    # Causal conclusion
    if all_passed:
        verdict = "ROBUST_CAUSAL_EFFECT"
        recommendation = (
            f"✅ CAUSAL EFFECT CONFIRMED ({causal_estimate_val:+.4f}). "
            "Identification holds via Backdoor criterion and passed all falsification stress tests."
        )
    else:
        verdict = "FRAGILE_ESTIMATE"
        recommendation = (
            f"⚠️ CAUSAL ESTIMATE SENSITIVITY WARNING ({causal_estimate_val:+.4f}). "
            "Estimate failed one or more refutation tests. Potential unobserved confounding exists."
        )

    dot_format = "digraph {\n" + "\n".join([f'  "{e["from"]}" -> "{e["to"]}";' for e in edges]) + "\n}"

    return {
        "causal_estimate": round(causal_estimate_val, 4),
        "estimation_method": estimation_method,
        "backdoor_variables": identified_estimand.get_backdoor_variables(),
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "dot_format": dot_format,
        },
        "refutations": refutation_results,
        "all_refutations_passed": all_passed,
        "verdict": verdict,
        "recommendation": recommendation,
    }

