"""
UI Component: Synthetic Control Method (SCM) & In-Space Placebos
Reference: Abadie, Diamond, Hainmueller (JASA 2010, AJPS 2015)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.synthetic_control import fit_synthetic_control, run_in_space_placebo_test


def render_synth_tab(df: pd.DataFrame):
    st.markdown("### 🌐 Synthetic Control Method (SCM) & In-Space Placebos")
    st.markdown(
        """
        **Advanced Quasi-Experimental Design:** When a feature is rolled out across an entire region (e.g. US-West or Germany), 
        finding a single matching control region is often impossible. 
        
        The **Synthetic Control Method (Abadie et al.)** uses convex quadratic optimization to construct a data-driven 
        synthetic counterfactual as a weighted combination of untreated donor regions:
        $$\\min_W \\| Y_{1, \\text{pre}} - Y_{0, \\text{pre}} W \\|_2^2 \\quad \\text{s.t.} \\quad w_j \\ge 0, \\; \\sum w_j = 1$$
        """
    )

    if df.empty:
        st.warning("Please upload a panel dataset or select a pre-loaded sample dataset in the sidebar.")
        return

    working_df = st.session_state.get("aggregated_panel_df", df)
    if "aggregated_panel_df" in st.session_state:
        st.success("⚡ Using dynamically aggregated cohort panel from your raw user dataset.")

    cols = list(working_df.columns)
    numeric_cols = [c for c in cols if np.issubdtype(working_df[c].dtype, np.number)]

    if not numeric_cols:
        st.info(
            "💡 **Continuous Outcome Required:** Synthetic Controls requires at least one numeric outcome metric. "
            "Please select **'🌍 Android Regional Rollout (DiD & SCM)'** in the sidebar."
        )
        return

    if "region" not in cols or ("week" not in cols and "date" not in cols):
        st.info(
            "💡 **Quasi-Experimental Panel Data Recommended:** Synthetic Control requires regional or cohort time series. "
            "To view an end-to-end Synthetic Control analysis with 12 global regions over 52 weeks, "
            "select **'🌍 Android Regional Rollout (DiD & SCM)'** in the sidebar."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        mapped_u = st.session_state.get("mapped_unit")
        default_unit = mapped_u if mapped_u in cols else ("region" if "region" in cols else cols[0])
        unit_col = st.selectbox(
            "Unit / Region Identifier",
            cols,
            index=cols.index(default_unit) if default_unit in cols else 0,
            key="scm_unit_col",
        )

    units = sorted(working_df[unit_col].dropna().unique())
    if len(units) < 3:
        st.error(f"Unit column '{unit_col}' has fewer than 3 units. Synthetic Control requires at least 1 treated unit and 2 donor units.")
        return

    if len(units) > 150:
        st.warning(
            f"The selected unit identifier '{unit_col}' contains {len(units):,} unique units (individual user IDs). "
            "Synthetic Control is designed for macro units (e.g., regions, states, markets)."
        )
        candidate_entities = [c for c in cols if 2 <= working_df[c].nunique() <= 100 and c not in numeric_cols]
        candidate_times = [c for c in cols if any(k in c.lower() for k in ["date", "time", "week", "day", "month"])]
        if candidate_entities and candidate_times:
            with st.expander("⚡ One-Click: Auto-Aggregate User Logs into a Regional / Cohort Panel", expanded=True):
                st.markdown(
                    "Transform your user-level events into a cohort panel (e.g. daily average metric per country or platform) "
                    "to immediately run Synthetic Control counterfactual optimization!"
                )
                acol1, acol2, acol3 = st.columns(3)
                with acol1:
                    agg_entity = st.selectbox("Cohort / Region Dimension", candidate_entities, key="scm_agg_entity")
                with acol2:
                    agg_time = st.selectbox("Time / Date Dimension", candidate_times, key="scm_agg_time")
                with acol3:
                    agg_outcome = st.selectbox("Metric to Average", numeric_cols, key="scm_agg_outcome")

                if st.button("🚀 Transform to Panel Dataset", key="scm_agg_btn", type="primary"):
                    from core.column_detector import aggregate_user_data_to_panel
                    treat_candidate = st.session_state.get("mapped_variant") or ("is_treatment" if "is_treatment" in cols else None)
                    panel_df = aggregate_user_data_to_panel(working_df, agg_entity, agg_time, agg_outcome, treat_candidate)
                    st.session_state["aggregated_panel_df"] = panel_df
                    st.success(f"Aggregated {len(working_df):,} user rows into {len(panel_df)} panel points! Reloading...")
                    st.rerun()
        return

    with col2:
        mapped_t = st.session_state.get("mapped_time")
        default_time = mapped_t if mapped_t in cols else ("week" if "week" in cols else ("date" if "date" in cols else cols[1]))
        time_col = st.selectbox(
            "Time / Period Column",
            cols,
            index=cols.index(default_time) if default_time in cols else 0,
            key="scm_time_col",
        )

    with col3:
        mapped_y = st.session_state.get("mapped_outcome")
        default_outcome = mapped_y if mapped_y in numeric_cols else ("engagement_score" if "engagement_score" in numeric_cols else numeric_cols[0])
        outcome_col = st.selectbox(
            "Outcome Metric",
            numeric_cols,
            index=numeric_cols.index(default_outcome) if default_outcome in numeric_cols else 0,
            key="scm_outcome_col",
        )

    sorted_times = sorted(working_df[time_col].dropna().unique())
    if len(sorted_times) < 4:
        st.warning(
            f"The selected time column '{time_col}' has only {len(sorted_times)} distinct periods. "
            "Synthetic Control requires at least 4 periods to estimate pre-intervention donor weights."
        )
        return

    col4, col5 = st.columns(2)
    with col4:
        default_treated = "US-West" if "US-West" in units else units[0]
        treated_unit = st.selectbox(
            "Treated Region / Unit",
            units,
            index=units.index(default_treated) if default_treated in units else 0,
            key="scm_treated_unit",
        )

    default_cutoff_idx = int(len(sorted_times) * 0.65) if len(sorted_times) > 3 else 0
    with col5:
        cutoff_time = st.select_slider(
            "Rollout Launch Time (t₀)",
            options=sorted_times,
            value=sorted_times[default_cutoff_idx],
            help="Select the exact period when the feature launched in the treated unit.",
            key="scm_cutoff_time",
        )

    try:
        scm_res = fit_synthetic_control(
            df=working_df,
            unit_col=unit_col,
            time_col=time_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            treatment_cutoff_time=cutoff_time,
        )
    except Exception as e:
        st.error(f"Synthetic Control Error: {e}")
        return

    fit_m = scm_res.get("fit_metrics", {})
    post_m = scm_res.get("post_metrics", {})

    def safe_scm_fmt(val, prefix="", decimals=3):
        if val is None or pd.isna(val):
            return "N/A"
        try:
            return f"{float(val):+{decimals}f}" if prefix == "+" else f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return str(val)

    # Metric Cards
    st.markdown("#### 🎯 Causal Lift & Synthetic Counterfactual Fit")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ate_val = post_m.get("average_treatment_effect")
        pct_lift = post_m.get("percentage_lift")
        st.metric(
            "Avg. Treatment Effect",
            safe_scm_fmt(ate_val, prefix="+"),
            delta=f"{pct_lift:+.2f}% Lift" if pct_lift is not None else None,
            help="Average gap between actual treated unit and synthetic counterfactual in post-treatment period."
        )
    with m2:
        st.metric("Cumulative Lift", safe_scm_fmt(post_m.get("cumulative_lift"), prefix="+", decimals=2))
    with m3:
        r2_val = fit_m.get("r_squared")
        st.metric(
            "Pre-Treatment RMSE",
            safe_scm_fmt(fit_m.get("pre_rmse")),
            delta=f"R² = {r2_val:.3f}" if r2_val is not None else None,
            help="Root Mean Squared Error of synthetic counterfactual fit during the pre-treatment period. Lower is better."
        )
    with m4:
        st.metric(
            "Post/Pre MSE Ratio",
            safe_scm_fmt(post_m.get("post_to_pre_mse_ratio"), decimals=2),
            help="Ratio of post-treatment variance to pre-treatment variance. Higher values indicate strong causal divergence."
        )

    # Visualizations: Trajectories and Gap
    st.markdown("#### 📈 Observed vs Synthetic Counterfactual")
    trajectories = scm_res["trajectories"]

    fig_scm = go.Figure()

    fig_scm.add_trace(go.Scatter(
        x=trajectories["time"], y=trajectories["treated_actual"],
        mode="lines+markers", line=dict(color="#1a73e8", width=3),
        name=f"Actual: {treated_unit}"
    ))

    fig_scm.add_trace(go.Scatter(
        x=trajectories["time"], y=trajectories["synthetic_counterfactual"],
        mode="lines", line=dict(color="#ea4335", width=3, dash="dash"),
        name="Synthetic Counterfactual"
    ))

    fig_scm.add_shape(
        type="line",
        x0=cutoff_time,
        x1=cutoff_time,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="black", dash="dash", width=2),
    )
    fig_scm.add_annotation(
        x=cutoff_time,
        y=1.02,
        yref="paper",
        text="Feature Launch (t₀)",
        showarrow=False,
        font=dict(size=11, color="black"),
    )

    fig_scm.update_layout(
        title=f"Synthetic Control Trajectory: {treated_unit} vs Synthetic Counterfactual",
        xaxis_title=time_col.capitalize(),
        yaxis_title=outcome_col.capitalize(),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=30, l=40, r=20),
    )
    st.plotly_chart(fig_scm, width="stretch")

    # Treatment Gap & Donor Weights
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown("##### 📉 Causal Effect Gap (Actual - Synthetic)")
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(
            x=trajectories["time"], y=trajectories["causal_gap"],
            mode="lines+markers", line=dict(color="#1e8e3e", width=2),
            marker=dict(size=6), name="Gap"
        ))
        fig_gap.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_gap.add_shape(
            type="line",
            x0=cutoff_time,
            x1=cutoff_time,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="black", dash="dash", width=2),
        )
        fig_gap.update_layout(
            title="Estimated Treatment Effect over Time",
            xaxis_title=time_col.capitalize(),
            yaxis_title=f"Gap in {outcome_col}",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(t=40, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_gap, width="stretch")

    with vcol2:
        st.markdown("##### ⚖️ Optimal Donor Unit Weights (W*)")
        weights_data = scm_res["donor_weights"]
        if weights_data:
            w_df = pd.DataFrame(weights_data)
            fig_w = go.Figure(go.Bar(
                x=w_df["weight_pct"], y=w_df["donor_unit"],
                orientation="h", marker_color="#1a73e8",
                text=[f"{v:.1f}%" for v in w_df["weight_pct"]], textposition="auto"
            ))
            fig_w.update_layout(
                title="Convex Weights Allocated to Donor Regions",
                xaxis_title="Weight Percentage (%)",
                yaxis_title="Donor Region",
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(t=40, b=30, l=80, r=20),
            )
            st.plotly_chart(fig_w, width="stretch")
        else:
            st.info("Weights evenly distributed across donors.")

    # Abadie In-Space Placebo Permutations
    st.markdown("#### 🔬 Abadie In-Space Placebo Permutation Tests")
    st.markdown(
        """
        **In-Space Placebo Permutation Test:** To evaluate if the observed lift could have arisen by pure chance, 
        we run the Synthetic Control algorithm iteratively on **every untreated donor region**. 
        If the treated unit's post-treatment divergence stands out compared to all donor placebos, the causal effect is statistically significant.
        """
    )

    if st.button("🚀 Run Placebo Permutation Test (All Donor Units)", type="primary", key="scm_placebo_btn"):
        with st.spinner("Computing in-space synthetic control models across donor units..."):
            placebo_res = run_in_space_placebo_test(
                df=working_df,
                unit_col=unit_col,
                time_col=time_col,
                outcome_col=outcome_col,
                treated_unit=treated_unit,
                treatment_cutoff_time=cutoff_time,
            )

        if placebo_res["status"] == "STATISTICALLY_SIGNIFICANT":
            st.success(placebo_res["recommendation"])
        else:
            st.warning(placebo_res["recommendation"])

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Exact Permutation p-value", f"{placebo_res['permutation_p_value']:.4f}")
        with p2:
            st.metric("Effect Divergence Rank", f"#{placebo_res['rank']} of {placebo_res['total_units_tested']}")
        with p3:
            st.metric("Treated Post/Pre Ratio", f"{placebo_res['treated_ratio']:.2f}")

        # The Abadie Spaghetti Plot
        placebo_df = placebo_res["placebo_gaps"]
        fig_spaghetti = go.Figure()

        # Add all placebo lines in semi-transparent grey
        for col in placebo_df.columns:
            if col not in ["time", treated_unit]:
                fig_spaghetti.add_trace(go.Scatter(
                    x=placebo_df["time"], y=placebo_df[col],
                    mode="lines", line=dict(color="rgba(160, 160, 160, 0.4)", width=1.5),
                    name="Donor Placebo", showlegend=False, hoverinfo="none"
                ))

        # Add treated unit line in bold blue
        fig_spaghetti.add_trace(go.Scatter(
            x=placebo_df["time"], y=placebo_df[treated_unit],
            mode="lines+markers", line=dict(color="#1a73e8", width=3.5),
            marker=dict(size=7), name=f"Treated: {treated_unit}"
        ))

        fig_spaghetti.add_hline(y=0, line_dash="dash", line_color="black")
        fig_spaghetti.add_shape(
            type="line",
            x0=cutoff_time,
            x1=cutoff_time,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="black", dash="dash", width=2),
        )
        fig_spaghetti.add_annotation(
            x=cutoff_time,
            y=1.02,
            yref="paper",
            text="Feature Launch",
            showarrow=False,
            font=dict(size=11, color="black"),
        )

        fig_spaghetti.update_layout(
            title="Abadie In-Space Placebos (Spaghetti Plot)",
            xaxis_title=time_col.capitalize(),
            yaxis_title=f"Gap in {outcome_col}",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_spaghetti, width="stretch")

