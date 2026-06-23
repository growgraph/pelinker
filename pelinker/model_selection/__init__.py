"""Model selection over embedding combinations (grid search + fusion)."""

from pelinker.model_selection.runner import run_model_selection
from pelinker.model_selection.summary import (
    SummaryFigureRenderResult,
    build_model_selection_summary_payload,
    render_model_selection_summary_figures,
)

__all__ = [
    "SummaryFigureRenderResult",
    "build_model_selection_summary_payload",
    "render_model_selection_summary_figures",
    "run_model_selection",
]
