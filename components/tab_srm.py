"""
UI Component: Sample Ratio Mismatch (SRM) Testing & Slice Diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.srm import calculate_srm, calculate_cumulative_srm, diagnose_srm_slices


def render_srm_tab(df: pd.DataFrame):
    st.markdown("### ⚖️ Automated Sample Ratio Mismatch (SRM) Testing")
    st.markdown(
        """
        **Why this matters at Google:** Before evaluating metrics or declaring winners, data scientists verify traffic integrity. 
        A Sample Ratio Mismatch occurs when the observed user split deviates significantly from the designed allocation (e.g., 50/50). 
        SRM is symptomatic of **severe engineering bugs**—such as redirect loops, browser crashes, caching mismatches, or telemetry drops—which invalidate the Average Treatment Effect (ATE).
        """
    )

    if df.empty:
        st.warning("Please upload a dataset or select a pre-loaded sample dataset in the sidebar.")
        return

    # Configuration controls
    cols = list(df.columns)
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    with col1:
        mapped_v = st.session_state.get("mapped_variant")
        default_variant_col = mapped_v if mapped_v in cols else ("variant" if "variant" in cols else cols[0])
        variant_col = st.selectbox(
            "Variant / Treatment Column",
            cols,
            index=cols.index(default_variant_col) if default_variant_col in cols else 0,
            key="srm_variant_col",
        )

    variants = sorted(df[variant_col].dropna().unique())
    if len(variants) < 2:
        st.error(f"Selected column '{variant_col}' has fewer than 2 distinct variants. Please choose a valid assignment column.")
        return

    with col2:
        alpha = st.selectbox(
            "Significance Threshold (α)",
            [0.01, 0.001, 0.05],
            index=0,
            help="Google standard practice uses α=0.01 or α=0.001 to prevent false alarms due to extreme sample sizes.",
            key="srm_alpha_select",
        )

    with col3:
        allocation_mode = st.radio(
            "Designed Split",
            ["Equal (50/50)", "Custom Ratios"],
            horizontal=True,
            key="srm_alloc_mode",
        )

    expected_ratios = {}
    if allocation_mode == "Equal (50/50)":
        for v in variants:
            expected_ratios[v] = 1.0 / len(variants)
    else:
        st.markdown("**Specify Expected Allocation Proportions:**")
        ratio_cols = st.columns(len(variants))
        total_ratio = 0.0
        for i, v in enumerate(variants):
            with ratio_cols[i]:
                r = st.number_input(f"Ratio: {v}", min_value=0.01, max_value=1.0, value=1.0 / len(variants), step=0.05, key=f"srm_ratio_{v}")
                expected_ratios[v] = r
                total_ratio += r
        # Normalize
        for v in variants:
            expected_ratios[v] /= total_ratio

    # Compute observed counts
    observed_counts = df[variant_col].value_counts().to_dict()
    for v in variants:
        if v not in observed_counts:
            observed_counts[v] = 0

    srm_result = calculate_srm(observed_counts, expected_ratios=expected_ratios, alpha=alpha)

    # Render Verdict Banner
    status = srm_result["status"]
    if status == "CRITICAL_SRM":
        st.error(srm_result["recommendation"])
    elif status == "WARNING_SRM":
        st.warning(srm_result["recommendation"])
    else:
        st.success(srm_result["recommendation"])

    # High-level Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Observed Users", f"{srm_result['total_sample_size']:,}")
    with m2:
        st.metric("Chi-Square Statistic (χ²)", f"{srm_result['chi2_stat']:.3f}")
    with m3:
        p_val_display = f"{srm_result['p_value']:.2e}" if srm_result['p_value'] < 0.001 else f"{srm_result['p_value']:.4f}"
        st.metric("p-value", p_val_display, delta="Pass" if srm_result['p_value'] >= alpha else "Fail", delta_color="normal" if srm_result['p_value'] >= alpha else "inverse")
    with m4:
        st.metric("Degrees of Freedom", srm_result["degrees_of_freedom"])

    # Detailed Variant Table
    st.markdown("#### 📋 Variant Allocation Breakdown")
    details_df = pd.DataFrame(srm_result["variant_details"])
    details_df.columns = [
        "Variant", "Observed Count", "Expected Count", "Observed Share",
        "Expected Share", "Count Delta", "% Deviation", "Std. Residual"
    ]
    st.dataframe(details_df.style.format({
        "Observed Count": "{:,}",
        "Expected Count": "{:,.1f}",
        "Observed Share": "{:.2%}",
        "Expected Share": "{:.2%}",
        "Count Delta": "{:+,}",
        "% Deviation": "{:+.2f}%",
        "Std. Residual": "{:+.2f}",
    }), width="stretch")

    # Visualizations: Observed vs Expected & Residuals
    st.markdown("#### 📊 Allocation Comparison & Residuals")
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        fig_alloc = go.Figure()
        obs_vals = [d["observed_count"] for d in srm_result["variant_details"]]
        exp_vals = [d["expected_count"] for d in srm_result["variant_details"]]
        var_names = [d["variant"] for d in srm_result["variant_details"]]

        fig_alloc.add_trace(go.Bar(
            x=var_names, y=obs_vals, name="Observed Traffic",
            marker_color="#1a73e8", text=[f"{v:,}" for v in obs_vals], textposition="auto"
        ))
        fig_alloc.add_trace(go.Bar(
            x=var_names, y=exp_vals, name="Designed Target",
            marker_color="#ea4335", opacity=0.6, text=[f"{v:,.0f}" for v in exp_vals], textposition="auto"
        ))
        fig_alloc.update_layout(
            title="Observed vs Designed Allocation",
            barmode="group",
            yaxis_title="User Count",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_alloc, width="stretch")

    with vcol2:
        residuals = [d["standardized_residual"] for d in srm_result["variant_details"]]
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=var_names, y=residuals,
            marker_color=["#d93025" if abs(r) > 2.58 else "#1e8e3e" for r in residuals],
            text=[f"{r:+.2f}σ" for r in residuals], textposition="auto"
        ))
        fig_res.add_hline(y=2.58, line_dash="dash", line_color="#d93025", annotation_text="+2.58σ (p=0.01)")
        fig_res.add_hline(y=-2.58, line_dash="dash", line_color="#d93025", annotation_text="-2.58σ (p=0.01)")
        fig_res.update_layout(
            title="Standardized Residuals (z-scores)",
            yaxis_title="Standard Deviations (σ)",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_res, width="stretch")

    # Cumulative Time Series Analysis (if Date column is available)
    date_candidates = [c for c in cols if any(k in c.lower() for k in ["date", "time", "day", "timestamp"])]
    if date_candidates:
        st.markdown("#### ⏱️ Cumulative Traffic Trajectory & SRM Evolution")
        mapped_t = st.session_state.get("mapped_time")
        default_date_idx = date_candidates.index(mapped_t) if mapped_t in date_candidates else 0
        date_col = st.selectbox(
            "Select Timestamp / Date Column",
            date_candidates,
            index=default_date_idx,
            key="srm_date_col",
        )
        try:
            cum_srm_df = calculate_cumulative_srm(df, date_col=date_col, variant_col=variant_col, expected_ratios=expected_ratios)
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=cum_srm_df["date"], y=cum_srm_df["p_value"],
                mode="lines+markers", line=dict(color="#1a73e8", width=3),
                marker=dict(size=7), name="Cumulative p-value"
            ))
            fig_time.add_hline(y=alpha, line_dash="dash", line_color="#ea4335", annotation_text=f"α = {alpha} Threshold")
            fig_time.update_layout(
                title="SRM p-value Trajectory Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Goodness-of-Fit p-value",
                yaxis_type="log",
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=50, b=30, l=40, r=20),
            )
            st.plotly_chart(fig_time, width="stretch")

            # Sequential Testing & Safe Continuous Monitoring
            with st.expander("🛡️ Safe Continuous Monitoring & Sequential Peeking Bounds (mSPRT / O'Brien-Fleming)", expanded=False):
                st.markdown(
                    "**The Peeking Problem at Google:** Checking fixed-horizon experiment p-values every day inflates the false positive rate "
                    "from 5% to over 30%. Sequential Alpha Spending boundaries (O'Brien-Fleming) adjust the critical threshold dynamically, "
                    "allowing teams to continuously monitor experiments and safely stop early for overwhelming wins or harm without alpha inflation."
                )
                from core.sequential import compute_sequential_bounds

                # Compute cumulative z-scores for variant 1
                v1_counts = cum_srm_df.get(f"count_{variants[0]}", cum_srm_df.iloc[:, 3])
                totals = cum_srm_df["cumulative_total"].values
                p_exp = expected_ratios.get(variants[0], 0.5)
                exp_counts = totals * p_exp
                se_counts = np.sqrt(np.maximum(totals * p_exp * (1.0 - p_exp), 1e-6))
                z_trajectory = (v1_counts.values - exp_counts) / se_counts

                seq_df = compute_sequential_bounds(
                    time_points=list(cum_srm_df["date"]),
                    z_scores=list(z_trajectory),
                    total_planned_periods=len(cum_srm_df),
                    alpha=alpha,
                    method="obrien_fleming",
                )

                fig_seq = go.Figure()
                fig_seq.add_trace(go.Scatter(
                    x=seq_df["checkpoint"], y=seq_df["upper_bound"],
                    mode="lines", line=dict(color="#ea4335", width=2, dash="dash"),
                    name="Upper Safe Stopping Boundary (+z)"
                ))
                fig_seq.add_trace(go.Scatter(
                    x=seq_df["checkpoint"], y=seq_df["lower_bound"],
                    mode="lines", line=dict(color="#ea4335", width=2, dash="dash"),
                    name="Lower Safe Stopping Boundary (-z)"
                ))
                fig_seq.add_trace(go.Scatter(
                    x=seq_df["checkpoint"], y=seq_df["observed_z_score"],
                    mode="lines+markers", line=dict(color="#1a73e8", width=3),
                    marker=dict(size=7), name=f"Cumulative z-score ({variants[0]})"
                ))
                fig_seq.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_seq.update_layout(
                    title="Sequential Stopping Boundaries (O'Brien-Fleming)",
                    xaxis_title="Date",
                    yaxis_title="Standardized Difference (z-score)",
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=50, b=30, l=40, r=20),
                )
                st.plotly_chart(fig_seq, width="stretch")
        except Exception as e:
            st.info(f"Could not calculate cumulative timeline: {e}")

    # Dimensional Slice Diagnostics
    st.markdown("#### 🔬 Root-Cause Slice Diagnostics")
    st.markdown(
        "Isolate which sub-population or dimension caused traffic leakage (e.g., iOS Safari redirect bug, bot traffic, regional proxy)."
    )

    categorical_candidates = [
        c for c in cols if c != variant_col and 2 <= df[c].nunique() <= 20 and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if not categorical_candidates:
        categorical_candidates = [c for c in cols if c != variant_col and 2 <= df[c].nunique() <= 20]

    if categorical_candidates:
        mapped_s = st.session_state.get("mapped_slices")
        default_slice_selection = [s for s in mapped_s if s in categorical_candidates] if mapped_s else categorical_candidates[:min(4, len(categorical_candidates))]
        if not default_slice_selection and categorical_candidates:
            default_slice_selection = categorical_candidates[:min(4, len(categorical_candidates))]
        selected_slices = st.multiselect(
            "Select Diagnostic Slicing Dimensions",
            categorical_candidates,
            default=default_slice_selection,
            key="srm_slice_cols",
        )
        if selected_slices:
            slice_df = diagnose_srm_slices(df, variant_col=variant_col, slice_cols=selected_slices, expected_ratios=expected_ratios)
            if not slice_df.empty:
                # Highlight anomalous slices
                st.dataframe(slice_df.style.apply(
                    lambda row: ["background-color: #fce8e6; color: #c5221f" if row["has_srm"] else "" for _ in row],
                    axis=1
                ).format({
                    "chi2_stat": "{:.2f}",
                    "p_value": "{:.2e}",
                }), width="stretch")
            else:
                st.info("No categorical slices had sufficient sample size for testing.")
    else:
        st.info("No categorical dimension columns detected for slice diagnostics.")

