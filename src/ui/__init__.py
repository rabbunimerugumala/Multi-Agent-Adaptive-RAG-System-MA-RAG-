"""Streamlit UI package for Multi-Agent Adaptive RAG."""

from src.ui.theme import apply_theme, toggle_theme, get_current_theme
from src.ui.components import (
    render_confidence_badge,
    render_evidence_panel,
    render_trace_panel,
    render_sources_panel,
    render_answer_card,
    render_comparison_card,
)

__all__ = [
    "apply_theme",
    "toggle_theme",
    "get_current_theme",
    "render_confidence_badge",
    "render_evidence_panel",
    "render_trace_panel",
    "render_sources_panel",
    "render_answer_card",
    "render_comparison_card",
]
