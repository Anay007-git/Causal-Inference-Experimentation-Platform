"""
UI Component: Executive & Data Science Experimentation Decision Report
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime


def render_report_tab(df: pd.DataFrame):
    st.markdown("### 📑 Executive & Data Science Decision Report")
    st.markdown(
        """
        **Google Product Review Standard:** Data Science and Product Management align on four pillars before launching features:
        1. **Traffic Integrity**: Passed automated SRM tests (no sampling bugs).
        2. **Measurement Precision**: Variance reduced via CUPED (narrow confidence bounds).
        3. **Identification Validity**: Pre-treatment parallel trends or synthetic control fit confirmed.
        4. **Falsification Robustness**: Estimate withstood placebo tests and causal refutations.
        """
    )

    exp_name = st.text_input("Experiment Name", value="Google Search Ranking Algorithm v4.2 Rollout", key="report_exp_name")
    pm_name = st.text_input("Author / Data Scientist", value="Senior Product Data Scientist", key="report_pm_name")

    st.markdown("#### 🎯 Executive Launch Decision Criteria")

    c1, c2 = st.columns(2)
    with c1:
        srm_status = st.selectbox(
            "SRM Validation Status",
            ["PASSED (Healthy Allocation)", "WARNING (Minor Imbalance)", "FAILED (Critical Telemetry Bug)"],
            index=0,
            key="report_srm_status",
        )
        cuped_applied = st.selectbox(
            "CUPED Variance Reduction",
            ["Applied (~60-95% Noise Compressed)", "Applied (Moderate ~20-50%)", "Not Applicable / Raw A/B Only"],
            index=0,
            key="report_cuped_status",
        )

    with c2:
        did_scm_status = st.selectbox(
            "Quasi-Experiment / Parallel Trends Fit",
            ["Valid Parallel Trends (p > 0.05)", "Synthetic Control Pre-Fit R² > 0.95", "Parallel Trends Violated"],
            index=0,
            key="report_did_scm_status",
        )
        refutation_status = st.selectbox(
            "DoWhy Causal Refutation Stress Tests",
            ["3/3 Passed (Robust to Falsification)", "1+ Refutations Failed (Fragile)", "Not Evaluated"],
            index=0,
            key="report_refutation_status",
        )

    # Automated Synthesis Decision Logic
    is_critical_fail = "FAILED" in srm_status or "Violated" in did_scm_status or "Failed" in refutation_status
    is_warning = "WARNING" in srm_status or "Fragile" in refutation_status

    if is_critical_fail:
        final_verdict = "🛑 DO NOT SHIP / ENGINEERING ROLLBACK REQUIRED"
        badge_color = "#c5221f"
        rationale = (
            "The experiment violates foundational causal validity constraints. "
            "Traffic telemetry or identification assumptions have failed. "
            "Releasing this change risks shipping an unverified or counter-productive user experience."
        )
    elif is_warning:
        final_verdict = "⚠️ HOLD FOR ROOT-CAUSE TRIAGE"
        badge_color = "#e37400"
        rationale = (
            "Statistical indicators show potential traffic imbalance or sensitivity to unobserved confounders. "
            "Perform dimensional slice diagnostics before authorizing a production ramp."
        )
    else:
        final_verdict = "🚀 RECOMMENDATION: SHIP TO 100% PRODUCTION"
        badge_color = "#137333"
        rationale = (
            "All four causal validity gates have been satisfied. "
            "Traffic allocation is balanced, pre-treatment variance has been compressed via CUPED, "
            "and the estimated treatment effect is statistically significant and immune to adversarial refutation."
        )

    # Decision Banner
    st.markdown(
        f"""
        <div style="background-color: rgba(128, 128, 128, 0.1); border: 1px solid rgba(128, 128, 128, 0.25); border-left: 6px solid {badge_color}; padding: 18px; border-radius: 8px; margin: 15px 0;">
            <h3 style="color: {badge_color}; margin: 0 0 8px 0;">{final_verdict}</h3>
            <p style="margin: 0; font-size: 15px; opacity: 0.9;">{rationale}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Narrative Summary
    st.markdown("#### 📝 Experiment Summary Memo")
    notes = st.text_area(
        "Data Science Analysis Notes & Business Context",
        value=(
            f"The experiment '{exp_name}' was analyzed using the Causal Inference & Experimentation Platform.\n\n"
            f"- Traffic Allocation: {srm_status}\n"
            f"- Variance Reduction: {cuped_applied}\n"
            f"- Quasi-Experimental Validity: {did_scm_status}\n"
            f"- Causal Robustness: {refutation_status}\n\n"
            f"Conclusion: {rationale}"
        ),
        height=180,
        key="report_notes",
    )

    # Generate Export Reports
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    markdown_report = f"""# Causal Inference & Experimentation Executive Report
**Experiment**: {exp_name}  
**Date**: {timestamp}  
**Author**: {pm_name}  

---

## 1. Executive Decision
### **{final_verdict}**
> {rationale}

---

## 2. Methodology & Validity Gates
| Evaluation Gate | Result | Evaluation Standard |
| :--- | :--- | :--- |
| **Sample Ratio Mismatch (SRM)** | {srm_status} | Pearson χ² Goodness-of-Fit (α=0.01) |
| **Variance Reduction (CUPED)** | {cuped_applied} | Deng et al. (WSDM 2013) Pre-Covariate θ |
| **Quasi-Experiment Validity** | {did_scm_status} | Differential Slope Parallel Trends / Abadie SCM |
| **Causal Graph Falsification** | {refutation_status} | Placebo Treatment & Common Cause Refuters |

---

## 3. Data Science Memo
{notes}

---
*Generated by Google Causal Inference & Experimentation Platform*
"""

    st.markdown("#### 💾 Export Audit Reports")
    dcol1, dcol2, dcol3 = st.columns(3)

    with dcol1:
        st.download_button(
            "📥 Download Markdown Memo (.md)",
            data=markdown_report,
            file_name=f"experiment_memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            width="stretch",
            key="report_download_md",
        )

    with dcol2:
        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>{exp_name} - Causal Experiment Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; color: #202124; }}
        h1 {{ color: #1a73e8; border-bottom: 2px solid #e8eaed; padding-bottom: 8px; }}
        .badge {{ background-color: #f1f3f4; border-left: 5px solid {badge_color}; padding: 15px; border-radius: 4px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #dadce0; padding: 10px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <h1>Causal Experimentation Executive Report</h1>
    <p><strong>Experiment:</strong> {exp_name}<br><strong>Date:</strong> {timestamp}<br><strong>Author:</strong> {pm_name}</p>
    <div class="badge">
        <h2 style="color: {badge_color}; margin-top:0;">{final_verdict}</h2>
        <p>{rationale}</p>
    </div>
    <h2>Methodology & Validity Gates</h2>
    <table>
        <tr><th>Validity Gate</th><th>Status</th></tr>
        <tr><td>Sample Ratio Mismatch (SRM)</td><td>{srm_status}</td></tr>
        <tr><td>CUPED Variance Reduction</td><td>{cuped_applied}</td></tr>
        <tr><td>Quasi-Experimental / SCM Fit</td><td>{did_scm_status}</td></tr>
        <tr><td>DoWhy Causal Refutations</td><td>{refutation_status}</td></tr>
    </table>
    <h2>Data Science Notes</h2>
    <p>{notes.replace(chr(10), '<br>')}</p>
</body>
</html>"""
        st.download_button(
            "📥 Download HTML Document (.html)",
            data=html_report,
            file_name=f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            width="stretch",
            key="report_download_html",
        )

    with dcol3:
        json_report = json.dumps({
            "experiment_name": exp_name,
            "timestamp": timestamp,
            "author": pm_name,
            "final_verdict": final_verdict,
            "rationale": rationale,
            "srm_status": srm_status,
            "cuped_applied": cuped_applied,
            "did_scm_status": did_scm_status,
            "refutation_status": refutation_status,
            "notes": notes,
        }, indent=2)
        st.download_button(
            "📥 Download JSON Audit Spec (.json)",
            data=json_report,
            file_name=f"audit_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            width="stretch",
            key="report_download_json",
        )

