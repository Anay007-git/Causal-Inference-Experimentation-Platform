"""
UI Component: CUPED Variance Reduction & Sample-Size Compression
Reference: Deng, Xu, Kohavi, Walker (WSDM 2013)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.cuped import apply_cuped, compute_sample_compression_savings


def render_cuped_tab(df: pd.DataFrame):
    st.markdown("### 📉 Variance Reduction via CUPED (Controlled-experiment Using Pre-Experiment Data)")
    st.markdown(
        """
        **The Google Data Science Standard:** Raw A/B tests suffer from high variance because users enter experiments with preexisting baseline variance. 
        CUPED removes this pre-treatment noise using historical user behavior (e.g., pre-experiment 14-day queries or watch time):
        $$\\tilde{Y} = Y - \\theta (X - \\mathbb{E}[X]), \\quad \\theta = \\frac{\\text{Cov}(Y, X)}{\\text{Var}(X)}$$
        Variance is reduced by **$\\rho^2$**, yielding tighter confidence intervals and enabling **massive sample-size compression** without sacrificing statistical power.
        """
    )

    if df.empty:
        st.warning("Please upload a dataset or select a pre-loaded sample dataset in the sidebar.")
        return

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) < 2:
        st.info(
            "💡 **Continuous Numeric Metrics Required:** CUPED requires at least one outcome metric (Y) and one baseline covariate (X). "
            "The current dataset has fewer than 2 continuous numeric metrics. "
            "Please select **'🔍 Google Search Ranking (A/B + CUPED)'** in the sidebar to evaluate variance reduction."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        mapped_v = st.session_state.get("mapped_variant")
        default_treat = mapped_v if mapped_v in cols else ("is_treatment" if "is_treatment" in cols else ("variant" if "variant" in cols else cols[0]))
        treat_col = st.selectbox(
            "Treatment Assignment Column",
            cols,
            index=cols.index(default_treat) if default_treat in cols else 0,
            key="cuped_treat_col",
        )

    unique_vals = sorted(df[treat_col].dropna().unique())
    if len(unique_vals) < 2:
        st.error(f"Treatment column '{treat_col}' must contain at least 2 distinct values.")
        return

    treat_val = unique_vals[1] if 1 in unique_vals or "treatment" in [str(u).lower() for u in unique_vals] else unique_vals[1]
    ctrl_val = unique_vals[0]

    with col2:
        mapped_y = st.session_state.get("mapped_outcome")
        default_outcome = mapped_y if mapped_y in numeric_cols else ("post_experiment_queries" if "post_experiment_queries" in numeric_cols else numeric_cols[-1])
        outcome_col = st.selectbox(
            "Post-Experiment Outcome Metric (Y)",
            numeric_cols,
            index=numeric_cols.index(default_outcome) if default_outcome in numeric_cols else 0,
            key="cuped_outcome_col",
        )

    with col3:
        remaining_num = [c for c in numeric_cols if c != outcome_col and c != treat_col]
        mapped_x = st.session_state.get("mapped_covariates") or []
        default_cov_selection = [c for c in mapped_x if c in remaining_num]
        if not default_cov_selection:
            default_cov = "pre_experiment_queries" if "pre_experiment_queries" in remaining_num else (remaining_num[0] if remaining_num else outcome_col)
            default_cov_selection = [default_cov] if default_cov in remaining_num else (remaining_num[:1] if remaining_num else [])

        covariate_cols = st.multiselect(
            "Pre-Experiment Baseline Covariates (X)",
            remaining_num,
            default=default_cov_selection,
            help="Select one or multiple baseline features to maximize variance reduction via OLS projection.",
            key="cuped_covariate_cols",
        )

    if not covariate_cols:
        st.info("Please select at least one pre-experiment covariate (X).")
        return

    alpha = st.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Default α = 0.05 corresponds to 95% Confidence Intervals.",
        key="cuped_alpha_slider",
    )

    try:
        cuped_res = apply_cuped(
            df=df,
            treatment_col=treat_col,
            outcome_col=outcome_col,
            covariate_col=covariate_cols,
            treatment_val=treat_val,
            control_val=ctrl_val,
            alpha=alpha,
        )
    except Exception as e:
        st.error(f"CUPED Calculation Error: {e}")
        return

    # Key Result Metrics Cards
    st.markdown("#### 🚀 Variance Reduction & Compression Gains")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Variance Reduction",
            f"{cuped_res['variance_reduction_pct']:.1f}%",
            delta=f"ρ = {cuped_res['correlation_rho']:.2f}",
            help="Percentage of statistical noise eliminated by adjusting for pre-experiment covariate."
        )
    with m2:
        st.metric(
            "Sample Size Multiplier",
            f"{cuped_res['sample_size_compression']:.2f}x",
            delta=f"-{cuped_res['traffic_savings_pct']:.1f}% traffic needed",
            help="1 / (1 - ρ²): Effective sample size compression multiplier."
        )
    with m3:
        st.metric(
            "Confidence Interval Narrowing",
            f"{cuped_res['cuped']['ci_reduction_pct']:.1f}%",
            delta="Tighter Bounds",
            help="Percentage reduction in CI width, boosting detection power for subtle effects."
        )
    with m4:
        th = cuped_res.get("theta")
        if isinstance(th, (int, float)):
            th_display = f"{th:.4f}"
        elif isinstance(th, list):
            th_display = f"[{', '.join(f'{x:.3f}' for x in th)}]"
        else:
            th_display = str(th)
        st.metric(
            "Optimal θ Weight",
            th_display,
            help="Optimal Cov(Y, X) / Var(X) adjustment weights."
        )

    # Comparison Grid: Raw vs CUPED
    st.markdown("#### ⚖️ Side-by-Side Impact Comparison")
    raw = cuped_res["raw"]
    cup = cuped_res["cuped"]

    comparison_df = pd.DataFrame([
        {
            "Analysis": "Standard Raw A/B",
            "Treated Mean": raw["treatment_mean"],
            "Control Mean": raw["control_mean"],
            "Lift (Δ)": raw["delta"],
            "% Lift": f"{raw['pct_lift']:+.2f}%",
            "Std. Error": raw["std_error"],
            "95% CI Lower": raw["ci_lower"],
            "95% CI Upper": raw["ci_upper"],
            "CI Width": raw["ci_width"],
            "p-value": f"{raw['p_value']:.4e}",
            "Stat. Sig (p<α)": "✅ Significant" if raw["is_significant"] else "❌ Not Sig",
        },
        {
            "Analysis": "CUPED Adjusted A/B",
            "Treated Mean": cup["treatment_mean"],
            "Control Mean": cup["control_mean"],
            "Lift (Δ)": cup["delta"],
            "% Lift": f"{cup['pct_lift']:+.2f}%",
            "Std. Error": cup["std_error"],
            "95% CI Lower": cup["ci_lower"],
            "95% CI Upper": cup["ci_upper"],
            "CI Width": cup["ci_width"],
            "p-value": f"{cup['p_value']:.4e}",
            "Stat. Sig (p<α)": "✅ Significant" if cup["is_significant"] else "❌ Not Sig",
        },
    ])
    st.dataframe(comparison_df.style.apply(
        lambda row: ["background-color: rgba(26, 115, 232, 0.22); font-weight: bold;" if row["Analysis"] == "CUPED Adjusted A/B" else "" for _ in row],
        axis=1
    ), width="stretch")

    # Visualizations
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown("##### 🌲 Confidence Interval Forest Plot")
        fig_ci = go.Figure()

        # Raw CI
        fig_ci.add_trace(go.Scatter(
            x=[raw["delta"]], y=["Standard Raw A/B"],
            mode="markers", marker=dict(size=12, color="#ea4335"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[raw["ci_upper"] - raw["delta"]],
                arrayminus=[raw["delta"] - raw["ci_lower"]],
                color="#ea4335", thickness=3, width=10
            ),
            name="Raw A/B CI"
        ))

        # CUPED CI
        fig_ci.add_trace(go.Scatter(
            x=[cup["delta"]], y=["CUPED Adjusted A/B"],
            mode="markers", marker=dict(size=12, color="#1a73e8"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[cup["ci_upper"] - cup["delta"]],
                arrayminus=[cup["delta"] - cup["ci_lower"]],
                color="#1a73e8", thickness=3, width=10
            ),
            name="CUPED CI"
        ))

        fig_ci.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Null Effect (Δ=0)")
        fig_ci.update_layout(
            title="95% Confidence Interval Comparison",
            xaxis_title="Treatment Effect (Absolute Lift)",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=40, b=30, l=120, r=20),
        )
        st.plotly_chart(fig_ci, width="stretch")

    with vcol2:
        st.markdown("##### 📊 Metric Distribution Variance Compression")
        adj_df = cuped_res["adjusted_data"]
        fig_dist = go.Figure()

        fig_dist.add_trace(go.Histogram(
            x=adj_df[outcome_col], name="Raw Y",
            opacity=0.45, marker_color="#ea4335", nbinsx=50
        ))
        fig_dist.add_trace(go.Histogram(
            x=adj_df["Y_cuped"], name="CUPED Ỹ",
            opacity=0.6, marker_color="#1a73e8", nbinsx=50
        ))
        fig_dist.update_layout(
            title="Distribution Spread: Raw vs Adjusted",
            barmode="overlay",
            xaxis_title="Metric Value",
            yaxis_title="User Frequency",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_dist, width="stretch")

    # Interactive Sample Size & Duration Compression Calculator
    st.markdown("#### 🧮 Interactive Sample-Size & Velocity Savings Calculator")
    st.markdown("Simulate the engineering acceleration your team unlocks using CUPED:")

    scol1, scol2 = st.columns(2)
    with scol1:
        orig_users = st.number_input(
            "Original Sample Size Required",
            min_value=1000,
            max_value=5000000,
            value=100000,
            step=10000,
            key="cuped_orig_users",
        )
    with scol2:
        orig_days = st.number_input(
            "Original Planned Experiment Runtime (Days)",
            min_value=1,
            max_value=90,
            value=14,
            step=1,
            key="cuped_orig_days",
        )

    savings = compute_sample_compression_savings(orig_users, orig_days, cuped_res["variance_reduction_pct"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("New Required Users", f"{savings['compressed_sample_size']:,}")
    with c2:
        st.metric("Users Saved", f"{savings['users_saved']:,}", delta=f"-{cuped_res['traffic_savings_pct']:.0f}%")
    with c3:
        st.metric("New Runtime Required", f"{savings['compressed_duration_days']} Days")
    with c4:
        st.metric("Days Saved", f"{savings['days_saved']} Days", delta=f"{savings['velocity_multiplier']}x Faster")

    # Pre-Experiment Power & MDE Sample Size Planner
    with st.expander("⚡ Pre-Experiment Power & MDE Sample Size Planner", expanded=False):
        st.markdown(
            "Determine the required sample size and runtime **before launching** based on your target Minimum Detectable Effect (MDE) and anticipated CUPED variance reduction:"
        )
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            target_mde = st.number_input(
                "Target MDE (% Relative Lift)",
                min_value=0.1,
                max_value=50.0,
                value=2.0,
                step=0.5,
                key="cuped_target_mde",
            )
        with pcol2:
            daily_traffic = st.number_input(
                "Estimated Daily Active Users",
                min_value=100,
                max_value=10000000,
                value=15000,
                step=1000,
                key="cuped_daily_traffic",
            )
        with pcol3:
            power_level = st.selectbox(
                "Statistical Power (1 - β)",
                [0.80, 0.90, 0.95],
                index=0,
                key="cuped_power_level",
            )

        from core.sequential import compute_power_and_sample_size
        clean_outcome = df[outcome_col].dropna()
        base_mean = float(clean_outcome.mean()) if not clean_outcome.empty else 100.0
        base_std = float(clean_outcome.std()) if not clean_outcome.empty else 20.0

        power_res = compute_power_and_sample_size(
            baseline_mean=base_mean,
            baseline_std=base_std,
            mde_relative_pct=target_mde,
            alpha=alpha,
            power=power_level,
            daily_traffic=daily_traffic,
            cuped_rho=cuped_res.get("correlation_rho", 0.0),
        )

        raw_plan = power_res["raw_experiment"]
        cup_plan = power_res["cuped_experiment"]

        plan_df = pd.DataFrame([
            {
                "Design": "Standard A/B Test",
                "Total Users Needed": f"{raw_plan['total_sample_size']:,}",
                "Users Per Variant": f"{raw_plan['sample_size_per_variant']:,}",
                "Duration (Days)": f"{raw_plan['estimated_duration_days']} Days",
                "Velocity Multiplier": "1.0x",
            },
            {
                "Design": f"CUPED Accelerated (ρ = {cup_plan['expected_rho']:.2f})",
                "Total Users Needed": f"{cup_plan['total_sample_size']:,}",
                "Users Per Variant": f"{cup_plan['sample_size_per_variant']:,}",
                "Duration (Days)": f"{cup_plan['estimated_duration_days']} Days",
                "Velocity Multiplier": f"{cup_plan['velocity_multiplier']:.2f}x Faster ({cup_plan['days_saved']} days saved)",
            },
        ])
        st.dataframe(plan_df, width="stretch")

