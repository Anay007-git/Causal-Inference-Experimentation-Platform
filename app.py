"""
Causal Inference & Experimentation Platform (A/B Testing & Synthetic Controls)
Designed for Product Data Science and Product Management Teams.

Tech Stack: Python, Streamlit, DoWhy, Statsmodels, SciPy, and Plotly.
Deployable on Vercel, Streamlit Community Cloud, Docker, and Google Cloud Run.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import core causal engine and dataset generators
from core.datasets import (
    generate_search_ranking_experiment,
    generate_srm_bug_dataset,
    generate_regional_rollout_data,
    generate_youtube_causal_data,
)
from core.column_detector import (
    detect_variant_column,
    detect_outcome_column,
    detect_baseline_covariate_columns,
    detect_unit_column,
    detect_time_column,
    detect_slice_columns,
)

# Import UI components
from components.tab_srm import render_srm_tab
from components.tab_cuped import render_cuped_tab
from components.tab_did import render_did_tab
from components.tab_synth import render_synth_tab
from components.tab_dowhy import render_dowhy_tab
from components.tab_report import render_report_tab
from components.tab_deployment import render_deployment_tab

# Set page config
st.set_page_config(
    page_title="Causal Inference & Experimentation Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Theme-Adaptive Google Material aesthetic (Supports Light & Dark Modes)
st.markdown(
    """
    <style>
    /* Google Clean Palette */
    :root {
        --primary-blue: #1a73e8;
        --google-red: #ea4335;
        --google-yellow: #fbbc04;
        --google-green: #34a853;
    }
    .main-header {
        font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        height: 100%;
    }
    .metric-card h4 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.05rem;
    }
    .metric-card p {
        font-size: 0.88rem;
        opacity: 0.85;
        margin: 0;
        line-height: 1.45;
    }

    /* High-Contrast Theme-Adaptive Tabs (Fully Legible in Dark and Light Themes) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid rgba(128, 128, 128, 0.25);
        padding-bottom: 4px;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        min-height: 44px;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        background-color: rgba(128, 128, 128, 0.14);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-bottom: none;
        transition: all 0.15s ease-in-out;
    }
    /* Force tab text to be high contrast and inherit Streamlit theme text color */
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: inherit !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(26, 115, 232, 0.22);
        border-color: #1a73e8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(26, 115, 232, 0.25) !important;
        border: 1px solid #1a73e8 !important;
        border-bottom: 3px solid #1a73e8 !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #1a73e8 !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🔬 Causal Inference & Experimentation Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production A/B Testing Validity, Sample-Size Compression via CUPED, Difference-in-Differences, Synthetic Controls, and DoWhy DAG Refutations</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration & Dataset Selector
st.sidebar.image("https://www.gstatic.com/images/branding/googlelogo/svg/googlelogo_clr_74x24px.svg", width=120)
st.sidebar.title("Data Source & Scenario")

dataset_choice = st.sidebar.selectbox(
    "Choose Experiment Scenario",
    [
        "🔍 Google Search Ranking (A/B + CUPED)",
        "🚨 Google Ads Checkout Bug (SRM Incident)",
        "🌍 Android Regional Rollout (DiD & SCM)",
        "📺 YouTube Recommendation (Causal DAG & Confounders)",
        "📁 Upload Custom Experiment CSV",
    ],
    index=0,
    key="sidebar_dataset_choice",
)

# Load data based on selection
@st.cache_data
def load_cached_dataset(choice: str):
    if "Google Search" in choice:
        return generate_search_ranking_experiment(n_users=35000, seed=42)
    elif "Google Ads" in choice:
        return generate_srm_bug_dataset(n_users=40000, seed=101)
    elif "Android Regional" in choice:
        return generate_regional_rollout_data(n_regions=12, n_weeks=52, treat_week=36, seed=777)
    elif "YouTube" in choice:
        return generate_youtube_causal_data(n_users=15000, seed=42)
    return pd.DataFrame()


current_df = pd.DataFrame()

if "Upload Custom" in dataset_choice:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Experiment Log", type=["csv"], key="sidebar_file_uploader")
    if uploaded_file is not None:
        try:
            current_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(current_df):,} rows and {len(current_df.columns)} columns.")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
    else:
        st.sidebar.info("Upload any experiment CSV or download template examples:")
        tcol1, tcol2 = st.sidebar.columns(2)
        with tcol1:
            sample_ab_csv = generate_search_ranking_experiment(500).to_csv(index=False)
            st.download_button(
                "📥 A/B + CUPED",
                data=sample_ab_csv,
                file_name="sample_ab_cuped.csv",
                mime="text/csv",
                width="stretch",
                key="sample_ab_csv_download",
            )
        with tcol2:
            sample_panel_csv = generate_regional_rollout_data(8, 24, 16).to_csv(index=False)
            st.download_button(
                "📥 Panel DiD",
                data=sample_panel_csv,
                file_name="sample_regional_panel.csv",
                mime="text/csv",
                width="stretch",
                key="sample_panel_csv_download",
            )
else:
    current_df = load_cached_dataset(dataset_choice)

# Schema Auto-Detection and Interactive Column Mapping
if not current_df.empty:
    all_cols = list(current_df.columns)
    num_cols = [c for c in all_cols if np.issubdtype(current_df[c].dtype, np.number)]

    auto_var = detect_variant_column(current_df)
    auto_out = detect_outcome_column(current_df, exclude_cols=[auto_var] if auto_var else [])
    auto_covs = detect_baseline_covariate_columns(current_df, outcome_col=auto_out, exclude_cols=[auto_var] if auto_var else [])
    auto_unit = detect_unit_column(current_df)
    auto_time = detect_time_column(current_df)
    auto_slices = detect_slice_columns(current_df, exclude_cols=[auto_var, auto_out] if auto_var and auto_out else [])

    with st.sidebar.expander("🎛️ Column Mapping & Schema Health", expanded=("Upload Custom" in dataset_choice)):
        st.caption("Review or customize detected experiment columns:")

        var_default_idx = all_cols.index(auto_var) if auto_var in all_cols else 0
        mapped_variant = st.selectbox("Variant / Arm Column", all_cols, index=var_default_idx, key="global_map_variant")
        st.session_state["mapped_variant"] = mapped_variant

        if num_cols:
            out_default_idx = num_cols.index(auto_out) if auto_out in num_cols else 0
            mapped_outcome = st.selectbox("Primary Metric (Y)", num_cols, index=out_default_idx, key="global_map_outcome")
            st.session_state["mapped_outcome"] = mapped_outcome

            avail_covs = [c for c in num_cols if c != mapped_outcome and c != mapped_variant]
            default_cov_selection = [c for c in auto_covs if c in avail_covs]
            mapped_covariates = st.multiselect("Pre-Baseline Covariates (X)", avail_covs, default=default_cov_selection, key="global_map_covariates")
            st.session_state["mapped_covariates"] = mapped_covariates
        else:
            st.session_state["mapped_outcome"] = None
            st.session_state["mapped_covariates"] = []
            st.info("No numeric columns detected in this dataset.")

        unit_default_idx = all_cols.index(auto_unit) if auto_unit in all_cols else 0
        mapped_unit = st.selectbox("Entity / Region Column", all_cols, index=unit_default_idx, key="global_map_unit")
        st.session_state["mapped_unit"] = mapped_unit

        time_options = ["None (Single Snapshot)"] + all_cols
        time_default_idx = time_options.index(auto_time) if auto_time in time_options else 0
        mapped_time_sel = st.selectbox("Time / Period Column", time_options, index=time_default_idx, key="global_map_time")
        st.session_state["mapped_time"] = None if mapped_time_sel == "None (Single Snapshot)" else mapped_time_sel
        st.session_state["mapped_slices"] = auto_slices

    with st.sidebar.expander("📊 Dataset Overview & Preview", expanded=False):
        st.write(f"**Total Records:** {len(current_df):,}")
        st.write(f"**Columns ({len(current_df.columns)}):**")
        st.code(", ".join(list(current_df.columns)[:10]))
        st.dataframe(current_df.head(5), width="stretch")

# Main Navigation Tabs
tab_overview, tab_srm, tab_cuped, tab_did, tab_synth, tab_dowhy, tab_report, tab_deploy = st.tabs([
    "📌 Platform Overview",
    "⚖️ 1. SRM Testing",
    "📉 2. CUPED Compression",
    "📊 3. Difference-in-Differences",
    "🌐 4. Synthetic Controls",
    "🕸️ 5. DoWhy Causal Graph",
    "📑 6. Executive Report",
    "🚀 7. Deployment Hub",
])

# Tab 0: Overview
with tab_overview:
    st.markdown("### 🏢 Google Product Data Science Experimentation Architecture")
    st.markdown(
        """
        Experimentation at Google and world-class product teams transcends basic two-sample t-tests. 
        When billions of user touchpoints and hundreds of millions in revenue depend on product launches, 
        statistical validity, traffic integrity, and quasi-experimental methods are non-negotiable.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color: #1a73e8; margin-top:0;">⚖️ Traffic Validity</h4>
                <p style="font-size: 13px; color:#5f6368;">Automated Sample Ratio Mismatch (SRM) checks with Pearson's χ² and exact binomial tests to catch telemetry and redirect bugs before evaluating metrics.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color: #188038; margin-top:0;">📉 Sample Compression</h4>
                <p style="font-size: 13px; color:#5f6368;">CUPED (Controlled-experiment Using Pre-Experiment Data) eliminates 50–90% of variance using baseline user history, drastically reducing experiment runtime.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color: #b06000; margin-top:0;">📊 Regional Quasi-Experiments</h4>
                <p style="font-size: 13px; color:#5f6368;">Difference-in-Differences (DiD) and Synthetic Controls (Abadie et al.) for feature rollouts where user-level randomization is impossible.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color: #d93025; margin-top:0;">🕸️ Causal Refutation</h4>
                <p style="font-size: 13px; color:#5f6368;">Structural Causal Modeling via DoWhy. Judea Pearl DAG identification, Backdoor adjustment, and adversarial falsification stress-tests.</p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### 🧭 Recommended Workflow for Data Scientists & PMs")

    st.markdown(
        """
        ```mermaid
        graph LR
            A["Upload Logs or Select Dataset"] --> B["Step 1: Check SRM (Traffic Integrity)"]
            B --> C{"SRM Passed?"}
            C -- "No (p < 0.01)" --> D["🚨 Halt! Run Slice Diagnostics & Fix Telemetry Bug"]
            C -- "Yes (p ≥ 0.01)" --> E{"Is Randomization Feasible?"}
            E -- "Yes (A/B Test)" --> F["Step 2: Apply CUPED Variance Reduction"]
            E -- "No (Regional Rollout)" --> G["Step 3: Run DiD & Synthetic Controls"]
            F --> H["Step 4: DoWhy Causal DAG & Falsification Refuters"]
            G --> H
            H --> I["Step 5: Generate Executive Decision Report (Ship / Hold)"]
            style A fill:#e8f0fe,stroke:#1a73e8,stroke-width:2px;
            style D fill:#fce8e6,stroke:#d93025,stroke-width:2px;
            style I fill:#e6f4ea,stroke:#188038,stroke-width:2px;
        ```
        """
    )

    st.markdown("#### 🚀 Quick-Start Scenarios")
    s1, s2 = st.columns(2)
    with s1:
        st.info(
            "👉 **Want to see CUPED in action?** Select **'🔍 Google Search Ranking (A/B + CUPED)'** in the sidebar and navigate to **Tab 2: CUPED Compression** to observe a ~95% variance reduction and 20x sample size compression."
        )
    with s2:
        st.warning(
            "👉 **Want to debug an SRM bug?** Select **'🚨 Google Ads Checkout Bug (SRM Incident)'** in the sidebar and navigate to **Tab 1: SRM Testing** to see the chi-square test trip and isolate the Safari Mobile bug in slice diagnostics."
        )

# Tab 1: SRM
with tab_srm:
    render_srm_tab(current_df)

# Tab 2: CUPED
with tab_cuped:
    render_cuped_tab(current_df)

# Tab 3: DiD
with tab_did:
    render_did_tab(current_df)

# Tab 4: Synthetic Controls
with tab_synth:
    render_synth_tab(current_df)

# Tab 5: DoWhy Causal Graph
with tab_dowhy:
    render_dowhy_tab(current_df)

# Tab 6: Executive Report
with tab_report:
    render_report_tab(current_df)

# Tab 7: Deployment Hub
with tab_deploy:
    render_deployment_tab()

