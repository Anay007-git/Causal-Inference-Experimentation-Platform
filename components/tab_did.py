"""
UI Component: Difference-in-Differences (DiD) & Dynamic Event Study
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.did import estimate_did, validate_parallel_trends, fit_event_study


def render_did_tab(df: pd.DataFrame):
    st.markdown("### 📊 Difference-in-Differences (DiD) Module")
    st.markdown(
        """
        **Quasi-Experimental Methods at Google:** When randomized controlled trials (RCTs) are unfeasible—such as regional marketing campaigns, 
        Android OS system-level upgrades, or regulatory rollouts affecting entire geographic markets—product teams use **Difference-in-Differences (DiD)**.
        
        DiD controls for permanent baseline differences between regions and macro time shocks affecting all units:
        $$Y_{it} = \\alpha + \\beta_1 \\text{Treat}_i + \\beta_2 \\text{Post}_t + \\beta_{\\text{DiD}} (\\text{Treat}_i \\times \\text{Post}_t) + \\varepsilon_{it}$$
        """
    )

    if df.empty:
        st.warning("Please upload a panel dataset or select a pre-loaded sample dataset in the sidebar.")
        return

    # Use aggregated panel if user generated one in session
    working_df = st.session_state.get("aggregated_panel_df", df)
    if "aggregated_panel_df" in st.session_state:
        st.success("⚡ Using dynamically aggregated cohort panel from your raw user dataset.")
        if st.button("Reset to Original Dataset", key="reset_did_panel"):
            del st.session_state["aggregated_panel_df"]
            st.rerun()

    cols = list(working_df.columns)
    numeric_cols = [c for c in cols if np.issubdtype(working_df[c].dtype, np.number)]

    if not numeric_cols:
        st.info(
            "💡 **Continuous Outcome Required:** Difference-in-Differences requires at least one numeric outcome metric. "
            "Please select **'🌍 Android Regional Rollout (DiD & SCM)'** in the sidebar."
        )
        return

    if "region" not in cols or ("week" not in cols and "date" not in cols):
        st.info(
            "💡 **Quasi-Experimental Panel Data Recommended:** Difference-in-Differences evaluates changes across "
            "regions or cohorts over time. To view an end-to-end DiD analysis with parallel trends validation "
            "and dynamic event studies across 12 global regions, select **'🌍 Android Regional Rollout (DiD & SCM)'** in the sidebar."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mapped_u = st.session_state.get("mapped_unit")
        default_unit = mapped_u if mapped_u in cols else ("region" if "region" in cols else cols[0])
        unit_col = st.selectbox(
            "Unit / Region Column",
            cols,
            index=cols.index(default_unit) if default_unit in cols else 0,
            key="did_unit_col",
        )

    if working_df[unit_col].nunique() > 150:
        st.warning(
            f"The selected unit column '{unit_col}' contains {working_df[unit_col].nunique():,} unique values (individual user IDs). "
            "DiD models regional markets or cohort aggregates rather than single users."
        )
        candidate_entities = [c for c in cols if 2 <= working_df[c].nunique() <= 100 and c not in numeric_cols]
        candidate_times = [c for c in cols if any(k in c.lower() for k in ["date", "time", "week", "day", "month"])]
        if candidate_entities and candidate_times:
            with st.expander("⚡ One-Click: Auto-Aggregate User Logs into a Regional / Cohort Panel", expanded=True):
                st.markdown(
                    "Transform your user-level events into a cohort panel (e.g. daily average metric per country or platform) "
                    "to immediately run Difference-in-Differences and Synthetic Controls!"
                )
                acol1, acol2, acol3 = st.columns(3)
                with acol1:
                    agg_entity = st.selectbox("Cohort / Region Dimension", candidate_entities, key="did_agg_entity")
                with acol2:
                    agg_time = st.selectbox("Time / Date Dimension", candidate_times, key="did_agg_time")
                with acol3:
                    agg_outcome = st.selectbox("Metric to Average", numeric_cols, key="did_agg_outcome")

                if st.button("🚀 Transform to Panel Dataset", key="did_agg_btn", type="primary"):
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
            "Time / Date Column",
            cols,
            index=cols.index(default_time) if default_time in cols else 0,
            key="did_time_col",
        )

    sorted_times = sorted(working_df[time_col].dropna().unique())
    if len(sorted_times) < 3:
        st.warning(
            f"The selected time column '{time_col}' has only {len(sorted_times)} distinct periods. "
            "Difference-in-Differences requires repeated time periods before and after the intervention."
        )
        return

    with col3:
        mapped_tr = st.session_state.get("mapped_variant")
        default_treat = mapped_tr if mapped_tr in cols else ("is_treated_region" if "is_treated_region" in cols else cols[2])
        treatment_col = st.selectbox(
            "Treatment Region Indicator",
            cols,
            index=cols.index(default_treat) if default_treat in cols else 0,
            key="did_treatment_col",
        )

    with col4:
        mapped_y = st.session_state.get("mapped_outcome")
        default_outcome = mapped_y if mapped_y in numeric_cols else ("engagement_score" if "engagement_score" in numeric_cols else numeric_cols[0])
        outcome_col = st.selectbox(
            "Outcome Metric",
            numeric_cols,
            index=numeric_cols.index(default_outcome) if default_outcome in numeric_cols else 0,
            key="did_outcome_col",
        )

    # Treatment cutoff time selector
    default_cutoff_idx = int(len(sorted_times) * 0.65) if len(sorted_times) > 3 else 0
    cutoff_time = st.select_slider(
        "Intervention / Rollout Launch Time (t₀)",
        options=sorted_times,
        value=sorted_times[default_cutoff_idx],
        help="Select the period when the feature was launched in the treated region.",
        key="did_cutoff_slider",
    )

    try:
        did_res = estimate_did(
            df=working_df,
            unit_col=unit_col,
            time_col=time_col,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            cutoff_time=cutoff_time,
            cov_type="HC3",
        )
    except Exception as e:
        st.error(f"DiD Estimation Error: {e}")
        return

    # Parallel Trends Test
    pt_res = validate_parallel_trends(
        df=working_df,
        time_col=time_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        cutoff_time=cutoff_time,
    )

    # Render Parallel Trends Assumption Banner
    if pt_res["status"] == "PASS":
        st.success(pt_res["recommendation"])
    elif pt_res["status"] == "FAIL":
        st.warning(pt_res["recommendation"])
    else:
        st.info(pt_res.get("message", "Parallel trends check skipped."))

    # Helper formatters to prevent NoneType formatting errors
    def safe_fmt(val, prefix="", suffix="", decimals=3):
        if val is None or pd.isna(val):
            return "N/A"
        try:
            return f"{float(val):+{decimals}f}" if prefix == "+" else f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return str(val)

    # DiD Metric Cards
    st.markdown("#### 🎯 Treatment Effect Estimates (ATT)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        did_coef = did_res.get("did_coefficient")
        pct_lift = did_res.get("pct_lift", 0.0)
        st.metric(
            "DiD Treatment Effect (β_DiD)",
            f"{did_coef:+.3f}" if did_coef is not None else "N/A",
            delta=f"{pct_lift:+.2f}% Lift" if pct_lift is not None else None,
            help="Average Treatment Effect on the Treated (ATT) after adjusting for time trends and unit fixed effects."
        )
    with m2:
        se_val = did_res.get("std_error")
        st.metric("Robust Std. Error", f"{se_val:.4f}" if se_val is not None else "N/A")
    with m3:
        p_val = did_res.get("p_value")
        p_val_display = (f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}") if p_val is not None else "N/A"
        st.metric(
            "p-value",
            p_val_display,
            delta="Significant" if did_res.get("is_significant") else "Not Significant",
            delta_color="normal" if did_res.get("is_significant") else "inverse"
        )
    with m4:
        ci_l = did_res.get("ci_lower")
        ci_u = did_res.get("ci_upper")
        ci_str = f"[{ci_l:+.3f}, {ci_u:+.3f}]" if (ci_l is not None and ci_u is not None) else "N/A"
        st.metric("95% Confidence Interval", ci_str)

    # 2x2 Empirical Means Grid
    st.markdown("#### 📐 2x2 Empirical Differences Table")
    m2x2 = did_res["means_2x2"]
    
    grid_df = pd.DataFrame([
        {
            "Cohort": "Treated Region",
            "Pre-Period Mean": safe_fmt(m2x2.get("treated_pre")),
            "Post-Period Mean": safe_fmt(m2x2.get("treated_post")),
            "Time Difference (Δ)": safe_fmt(m2x2.get("treated_delta"), prefix="+"),
        },
        {
            "Cohort": "Control Regions (Avg)",
            "Pre-Period Mean": safe_fmt(m2x2.get("control_pre")),
            "Post-Period Mean": safe_fmt(m2x2.get("control_post")),
            "Time Difference (Δ)": safe_fmt(m2x2.get("control_delta"), prefix="+"),
        },
        {
            "Cohort": "Difference-in-Differences (DiD)",
            "Pre-Period Mean": "-",
            "Post-Period Mean": "-",
            "Time Difference (Δ)": safe_fmt(m2x2.get("did_empirical"), prefix="+"),
        },
    ])
    st.dataframe(grid_df.style.apply(
        lambda row: ["background-color: #e8f0fe; font-weight: bold;" if row["Cohort"] == "Difference-in-Differences (DiD)" else "" for _ in row],
        axis=1
    ), width="stretch")

    # Trajectory Plot with Counterfactual
    st.markdown("#### 📈 Observed vs Counterfactual Trajectories")
    traj = did_res["trajectories"]

    fig_traj = go.Figure()

    if "Treated" in traj.columns:
        fig_traj.add_trace(go.Scatter(
            x=traj[time_col], y=traj["Treated"],
            mode="lines+markers", line=dict(color="#1a73e8", width=3),
            name="Treated Region (Actual)"
        ))

    if "Control" in traj.columns:
        fig_traj.add_trace(go.Scatter(
            x=traj[time_col], y=traj["Control"],
            mode="lines", line=dict(color="#ea4335", width=2, dash="dot"),
            name="Control Regions (Average)"
        ))

    if "Counterfactual" in traj.columns and not traj["Counterfactual"].isna().all():
        fig_traj.add_trace(go.Scatter(
            x=traj[time_col], y=traj["Counterfactual"],
            mode="lines", line=dict(color="#f9ab00", width=3, dash="dash"),
            name="Counterfactual (If Unchanged)"
        ))

    # Add vertical line for launch cutoff
    fig_traj.add_shape(
        type="line",
        x0=cutoff_time,
        x1=cutoff_time,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="black", dash="dash", width=2),
    )
    fig_traj.add_annotation(
        x=cutoff_time,
        y=1.02,
        yref="paper",
        text="Feature Launch (t₀)",
        showarrow=False,
        font=dict(size=11, color="black"),
    )

    fig_traj.update_layout(
        title="DiD Trajectories: Actual vs Counterfactual Projection",
        xaxis_title=time_col.capitalize(),
        yaxis_title=outcome_col.capitalize(),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=30, l=40, r=20),
    )
    st.plotly_chart(fig_traj, width="stretch")

    # Dynamic Event Study Leads & Lags
    st.markdown("#### ⏱️ Dynamic Event Study Specification (Leads & Lags)")
    st.markdown(
        "Event study regressions inspect whether pre-treatment estimates are centered at zero (validating parallel trends) "
        "and reveal whether the treatment effect grows, fades, or persists over post-launch periods."
    )

    try:
        event_df = fit_event_study(
            df=working_df,
            time_col=time_col,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            cutoff_time=cutoff_time,
        )

        fig_event = go.Figure()

        # Pre-periods in gray/orange, post-periods in blue
        colors = ["#f2994a" if row["is_pre_period"] else "#1a73e8" for _, row in event_df.iterrows()]

        fig_event.add_trace(go.Scatter(
            x=event_df["period_label"],
            y=event_df["estimate"],
            mode="lines+markers",
            line=dict(color="#1a73e8", width=2),
            marker=dict(size=8, color=colors),
            error_y=dict(
                type="data",
                symmetric=False,
                array=event_df["ci_upper"] - event_df["estimate"],
                arrayminus=event_df["estimate"] - event_df["ci_lower"],
                color="#5f6368", thickness=2, width=6
            ),
            name="Point Estimate ± 95% CI"
        ))

        fig_event.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Null (Effect = 0)")
        fig_event.update_layout(
            title="Event Study Coefficients around Rollout Date",
            xaxis_title="Relative Periods to Feature Launch",
            yaxis_title="Estimated Effect (β_k)",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=30, l=40, r=20),
        )
        st.plotly_chart(fig_event, width="stretch")
    except Exception as e:
        st.info(f"Could not fit event study model: {e}")

    # Full Regression Summary
    with st.expander("🔍 View Full OLS Regression Summary"):
        st.code(did_res["regression_summary"])
