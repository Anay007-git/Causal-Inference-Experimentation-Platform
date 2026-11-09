"""
UI Components Package
"""

from components.tab_srm import render_srm_tab
from components.tab_cuped import render_cuped_tab
from components.tab_did import render_did_tab
from components.tab_synth import render_synth_tab
from components.tab_dowhy import render_dowhy_tab
from components.tab_report import render_report_tab
from components.tab_deployment import render_deployment_tab

__all__ = [
    "render_srm_tab",
    "render_cuped_tab",
    "render_did_tab",
    "render_synth_tab",
    "render_dowhy_tab",
    "render_report_tab",
    "render_deployment_tab",
]

