"""UI Component library for Multi-Agent Adaptive RAG.

Renders custom styled cards, confidence badges, agent execution timelines,
and expandable inspection panels with clean SVG icons (zero emojis).
"""

from typing import List, Dict, Any, Optional
import streamlit as st
from src.ui.icons import icon_svg


def get_confidence_badge_html(confidence: float) -> str:
    """Generates HTML markup for pill-shaped confidence badge with SVG dot."""
    pct = round(confidence * 100, 1)
    if confidence >= 0.8:
        badge_class = "badge-high"
        dot_color = "#10B981"
        tier = "High Confidence"
    elif confidence >= 0.5:
        badge_class = "badge-medium"
        dot_color = "#F59E0B"
        tier = "Medium Confidence"
    else:
        badge_class = "badge-low"
        dot_color = "#EF4444"
        tier = "Low Confidence"

    dot_icon = icon_svg("dot", size=14, color=dot_color)
    return f"""<span class="badge-pill {badge_class}">{dot_icon}<span>{pct}% {tier}</span></span>"""


def render_confidence_badge(confidence: float):
    """Renders confidence pill badge inside Streamlit container."""
    st.markdown(get_confidence_badge_html(confidence), unsafe_allow_html=True)


def render_answer_card(
    answer: str,
    confidence: float,
    route: str = "complex",
    retry_count: int = 0,
    is_standard_rag: bool = False,
):
    """Renders the primary answer card with badges and metadata."""
    badge_html = get_confidence_badge_html(confidence)
    title_label = "Standard RAG Baseline" if is_standard_rag else "Multi-Agent Verified Answer"
    icon = icon_svg("cpu", size=18, color="#818CF8") if is_standard_rag else icon_svg("shield_check", size=18, color="#10B981")
    
    route_icon = icon_svg("layers", size=13, color="#818CF8")
    route_badge = (
        f'<span class="badge-route">{route_icon}<span>Route: {route.upper()}</span></span>'
        if not is_standard_rag
        else f'<span class="badge-route" style="background:#374151;color:#E5E7EB;">{route_icon}<span>Mode: Baseline</span></span>'
    )
    
    retry_icon = icon_svg("refresh", size=13, color="#F59E0B")
    retry_badge = (
        f'<span class="badge-retry">{retry_icon}<span>Retries: {retry_count}</span></span>'
        if retry_count > 0
        else ""
    )

    card_html = f"""
    <div class="answer-card">
        <div class="answer-card-header">
            <div class="answer-card-title">
                {icon}
                <span>{title_label}</span>
                {route_badge}
                {retry_badge}
            </div>
            <div>
                {badge_html}
            </div>
        </div>
        <div class="answer-card-body">
            {answer}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_evidence_panel(
    evidence: List[str], evidence_sources: Optional[List[Dict[str, Any]]] = None
):
    """Renders the expandable Evidence panel with clean file icons."""
    panel_title = "Retrieved Evidence Chunks"
    with st.expander(panel_title, expanded=False):
        if not evidence:
            st.info("No evidence chunks available.")
            return

        for idx, text in enumerate(evidence):
            source_info = (
                evidence_sources[idx]
                if (evidence_sources and idx < len(evidence_sources))
                else {}
            )
            title = source_info.get("title", f"Document #{idx+1}")
            score = source_info.get("score", 0.0)

            score_badge = (
                f'<span class="evidence-score">Similarity: {score * 100:.1f}%</span>'
                if score > 0
                else ""
            )

            file_icon = icon_svg("file_text", size=15, color="#818CF8")
            chunk_html = f"""
            <div class="evidence-card">
                <div class="evidence-header">
                    <span style="display:inline-flex; align-items:center; gap:6px;">{file_icon} <b>{title}</b> (Chunk #{idx+1})</span>
                    {score_badge}
                </div>
                <div class="evidence-body">
                    {text}
                </div>
            </div>
            """
            st.markdown(chunk_html, unsafe_allow_html=True)


def render_trace_panel(trace: List[Dict[str, Any]]):
    """Renders the expandable Agent Execution Trace panel."""
    panel_title = "Agent Execution Trace & Audit Log"
    with st.expander(panel_title, expanded=False):
        if not trace:
            st.info("No trace entries recorded.")
            return

        st.markdown('<div class="trace-timeline">', unsafe_allow_html=True)
        for idx, step in enumerate(trace):
            agent = step.get("agent", "Unknown Agent")
            action = step.get("action", "")
            timestamp = step.get("timestamp", "")
            details = step.get("details", {})

            detail_lines = []
            for k, v in details.items():
                if isinstance(v, list):
                    v_str = ", ".join(str(item) for item in v[:5])
                    if len(v) > 5:
                        v_str += f" (+{len(v)-5} more)"
                else:
                    v_str = str(v)
                detail_lines.append(f"<b>{k}:</b> {v_str}")
            details_formatted = " | ".join(detail_lines)

            pulse_icon = icon_svg("pulse", size=14, color="#6366F1")
            item_html = f"""
            <div class="trace-item">
                <div class="trace-item-header">
                    <span style="display:inline-flex; align-items:center; gap:6px;">{pulse_icon} Step {idx+1}: {agent}</span>
                    <span class="trace-item-time">{timestamp}</span>
                </div>
                <div style="font-weight: 500; margin-bottom: 4px; color: #E5E7EB;">{action}</div>
                <div class="trace-item-detail">{details_formatted}</div>
            </div>
            """
            st.markdown(item_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_sources_panel(evidence_sources: List[Dict[str, Any]]):
    """Renders the expandable Sources panel with document metadata."""
    panel_title = "Document Sources & Citations"
    with st.expander(panel_title, expanded=False):
        if not evidence_sources:
            st.info("No sources recorded for this query.")
            return

        unique_sources = {}
        for s in evidence_sources:
            t = s.get("title", "Unknown")
            if t not in unique_sources:
                unique_sources[t] = s

        link_icon = icon_svg("link", size=14, color="#818CF8")
        for title, info in unique_sources.items():
            score = info.get("score", 0.0)
            meta = info.get("metadata", {})
            st.markdown(
                f'<div style="margin-bottom:6px; font-size:0.9rem;">{link_icon} <b>{title}</b> &mdash; Relevance: <code>{score * 100:.1f}%</code> | Source: <code>{meta.get("source", "HotpotQA")}</code></div>',
                unsafe_allow_html=True,
            )


def render_comparison_card(sample: Dict[str, Any], index: int = 1):
    """Renders a side-by-side comparison between Standard RAG and Multi-Agent RAG with SVG badges."""
    question = sample.get("question", "")
    ground_truth = sample.get("ground_truth", "")
    std = sample.get("standard_rag", {})
    marag = sample.get("marag", {})

    std_correct = std.get("correct", False)
    marag_correct = marag.get("correct", False)

    std_status_icon = icon_svg("check_circle", size=18, color="#10B981") if std_correct else icon_svg("x_circle", size=18, color="#EF4444")
    std_status_text = '<span style="color:#10B981; font-weight:600;">Accurate</span>' if std_correct else '<span style="color:#EF4444; font-weight:600;">Hallucinated / Incomplete</span>'

    marag_status_icon = icon_svg("check_circle", size=18, color="#10B981") if marag_correct else icon_svg("x_circle", size=18, color="#EF4444")
    marag_status_text = '<span style="color:#10B981; font-weight:600;">Accurate</span>' if marag_correct else '<span style="color:#EF4444; font-weight:600;">Incomplete</span>'

    marag_conf = marag.get("confidence", 0.0)
    badge_html = get_confidence_badge_html(marag_conf)
    target_icon = icon_svg("target", size=16, color="#6366F1")

    st.markdown(
        f"""
        <div style="border: 1px solid #1F2937; border-radius: 12px; padding: 18px; margin-bottom: 16px; background: #111827;">
            <div style="font-weight: 700; font-size: 1.02rem; margin-bottom: 8px; color:#F9FAFB;">
                Sample {index}: {question}
            </div>
            <div style="font-size: 0.88rem; color: #9CA3AF; display:flex; align-items:center; gap:8px;">
                {target_icon}
                <span><b>Ground Truth Answer:</b> <span style="color: #818CF8; font-weight: 600;">{ground_truth}</span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        cpu_icon = icon_svg("cpu", size=16, color="#94A3B8")
        st.markdown(
            f"""
            <div class="comparison-box standard">
                <div class="comparison-header">
                    <span style="display:inline-flex; align-items:center; gap:6px;">{cpu_icon} <span>Standard RAG</span></span>
                    <span style="display:inline-flex; align-items:center; gap:4px;">{std_status_icon} {std_status_text}</span>
                </div>
                <div style="font-size: 0.94rem; margin-bottom: 12px; min-height: 70px; color:#D1D5DB; line-height:1.55;">
                    {std.get("answer", "")}
                </div>
                <div style="font-size: 0.8rem; color: #9CA3AF; border-top: 1px solid #1F2937; padding-top: 8px;">
                    Self-Verification: Disabled | Retries: 0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        shield_icon = icon_svg("shield_check", size=16, color="#10B981")
        st.markdown(
            f"""
            <div class="comparison-box marag">
                <div class="comparison-header">
                    <span style="display:inline-flex; align-items:center; gap:6px;">{shield_icon} <span>Multi-Agent RAG</span></span>
                    <span style="display:inline-flex; align-items:center; gap:4px;">{marag_status_icon} {marag_status_text}</span>
                </div>
                <div style="font-size: 0.94rem; margin-bottom: 12px; min-height: 70px; color:#D1D5DB; line-height:1.55;">
                    {marag.get("answer", "")}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; border-top: 1px solid #1F2937; padding-top: 8px;">
                    {badge_html}
                    <span style="font-size:0.8rem; color:#9CA3AF;">Adaptive Loops: {marag.get('retry_count', 0)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    trace = marag.get("trace", [])
    if trace:
        with st.expander(f"View Agent Execution Trace (Sample {index})", expanded=False):
            render_trace_panel(trace)
