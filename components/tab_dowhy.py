"""
UI Component: Causal Graph Modeling & Refutation Engine (DoWhy)
Reference: Sharma, Kiciman (Microsoft Research, 2020)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.causal_graph import run_dowhy_causal_analysis


def render_dowhy_tab(df: pd.DataFrame):
    st.markdown("### 🕸️ Causal Graph Modeling & Refutation Engine (DoWhy)")
    st.markdown(
        """
        **Moving Beyond Correlation to Causality:** Data science teams encounter observational data where randomization was impossible. 
        **DoWhy** enforces Judea Pearl's structural causal framework through four rigorous steps:
        1. **Model**: Explicitly define the Directed Acyclic Graph (DAG) of treatments, outcomes, and confounders.
        2. **Identify**: Check if the causal effect is identifiable via the Backdoor Criterion.
        3. **Estimate**: Fit the identified causal estimand using statistical or ML estimators.
        4. **Refute**: Subject the estimate to **falsification stress-tests** (placebo treatments, random common causes, and subset bootstrapping).
        """
    )

    if df.empty:
        st.warning("Please upload a dataset or select a pre-loaded sample dataset in the sidebar.")
        return

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        st.info(
            "💡 **Continuous Outcome Metric Required:** Causal graph identification requires a numeric outcome variable (Y). "
            "Please select **'📺 YouTube Recommendation'** or **'🔍 Google Search Ranking'** in the sidebar."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        mapped_tr = st.session_state.get("mapped_variant")
        default_treat = mapped_tr if mapped_tr in cols else ("recommendation_v2" if "recommendation_v2" in cols else ("is_treatment" if "is_treatment" in cols else cols[0]))
        treatment_col = st.selectbox(
            "Treatment Variable (T)",
            cols,
            index=cols.index(default_treat) if default_treat in cols else 0,
            key="dowhy_treatment_col",
        )

    with col2:
        mapped_y = st.session_state.get("mapped_outcome")
        default_outcome = mapped_y if mapped_y in numeric_cols else ("watch_time_hours" if "watch_time_hours" in numeric_cols else numeric_cols[-1])
        outcome_col = st.selectbox(
            "Outcome Variable (Y)",
            numeric_cols,
            index=numeric_cols.index(default_outcome) if default_outcome in numeric_cols else 0,
            key="dowhy_outcome_col",
        )

    remaining_cols = [c for c in cols if c not in [treatment_col, outcome_col]]

    col3, col4 = st.columns(2)
    with col3:
        default_confounders = [c for c in ["tenure_months", "network_speed_mbps", "screen_category"] if c in remaining_cols]
        common_causes = st.multiselect(
            "Confounders / Common Causes (W)",
            remaining_cols,
            default=default_confounders if default_confounders else remaining_cols[:min(3, len(remaining_cols))],
            key="dowhy_confounders",
        )

    with col4:
        estimation_method = st.selectbox(
            "Estimation Method",
            ["backdoor.linear_regression", "backdoor.propensity_score_stratification"],
            index=0,
            help="Statistical estimator to isolate the true causal effect after conditioning on confounders.",
            key="dowhy_estimation_method",
        )

    if st.button("🚀 Run Full Causal Identification & Refutation Pipeline", type="primary", key="dowhy_run_btn"):
        with st.spinner("Building causal DAG, identifying Backdoor estimand, estimating effect, and running 3 falsification refuters..."):
            try:
                causal_res = run_dowhy_causal_analysis(
                    df=df,
                    treatment=treatment_col,
                    outcome=outcome_col,
                    common_causes=common_causes,
                    estimation_method=estimation_method,
                    run_refutations=True,
                )
            except Exception as e:
                st.error(f"DoWhy Execution Error: {e}")
                return

        # Verdict Banner
        if causal_res["all_refutations_passed"]:
            st.success(causal_res["recommendation"])
        else:
            st.warning(causal_res["recommendation"])

        # Metric Summary Cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Estimated Causal Effect (ATE)",
                f"{causal_res['causal_estimate']:+.4f}",
                help="True expected change in outcome caused exclusively by treatment intervention."
            )
        with m2:
            st.metric(
                "Backdoor Variables",
                len(causal_res["backdoor_variables"]),
                help="Set of confounding variables conditioned upon to block non-causal backdoor paths."
            )
        with m3:
            st.metric(
                "Falsification Tests Passed",
                "3 / 3" if causal_res["all_refutations_passed"] else "Failed Checks",
                delta="Robust" if causal_res["all_refutations_passed"] else "Fragile",
                delta_color="normal" if causal_res["all_refutations_passed"] else "inverse"
            )

        # Causal DAG Visualizer
        st.markdown("#### 🗺️ Identified Causal Graph (DAG)")
        nodes = causal_res["graph"]["nodes"]
        edges = causal_res["graph"]["edges"]

        # Render DAG as clean Mermaid markdown
        mermaid_code = "graph LR\n"
        for n in nodes:
            if n["type"] == "treatment":
                mermaid_code += f'    {n["id"]}["🔵 Treatment: {n["id"]}"]:::treat\n'
            elif n["type"] == "outcome":
                mermaid_code += f'    {n["id"]}["🟢 Outcome: {n["id"]}"]:::out\n'
            else:
                mermaid_code += f'    {n["id"]}["🟠 Confounder: {n["id"]}"]:::conf\n'

        for e in edges:
            mermaid_code += f'    {e["from"]} --> {e["to"]}\n'

        mermaid_code += "    classDef treat fill:#e8f0fe,stroke:#1a73e8,stroke-width:2px,color:#1a73e8;\n"
        mermaid_code += "    classDef out fill:#e6f4ea,stroke:#137333,stroke-width:2px,color:#137333;\n"
        mermaid_code += "    classDef conf fill:#fef7e0,stroke:#b06000,stroke-width:2px,color:#b06000;\n"

        st.markdown(f"```mermaid\n{mermaid_code}\n```")

        # Refutation Stress-Test Results
        st.markdown("#### 🛡️ Falsification & Robustness Stress Tests")
        st.markdown(
            "Causal claims must be falsifiable. The platform subjected the estimate to three adversarial refutation attacks:"
        )

        for ref in causal_res["refutations"]:
            with st.container():
                rcol1, rcol2, rcol3 = st.columns([3, 1.5, 1.5])
                with rcol1:
                    st.markdown(f"**{ref['test_name']}**")
                    st.caption(ref.get("description", ""))
                    st.markdown(ref.get("summary", ""))
                with rcol2:
                    st.metric("Original Estimate", f"{ref.get('original_effect', 0.0):+.4f}")
                with rcol3:
                    st.metric(
                        "Refuted Estimate",
                        f"{ref.get('refuted_effect', 0.0):+.4f}",
                        delta="Passed" if ref.get("passed", False) else "Failed",
                        delta_color="normal" if ref.get("passed", False) else "inverse"
                    )
                st.divider()

