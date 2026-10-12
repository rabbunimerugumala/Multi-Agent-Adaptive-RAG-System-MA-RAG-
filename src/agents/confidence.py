"""Confidence Agent module for Multi-Agent Adaptive RAG.

Calculates the mathematically calibrated composite confidence score
(0.6 * verifier_score + 0.4 * avg_evidence_score) and controls whether
the system approves the answer or triggers an adaptive re-retrieval loop.
"""

from datetime import datetime
from typing import Dict, Any

from config import config, setup_logger
from src.state import AgentState

logger = setup_logger("confidence_agent")


def calculate_composite_confidence(
    verifier_score: float, avg_evidence_score: float
) -> float:
    """Computes calibrated weighted confidence metric.

    Formula:
        confidence = 0.6 * verifier_score + 0.4 * avg_evidence_score

    Args:
        verifier_score: Score from Verifier Agent (0.0 to 1.0).
        avg_evidence_score: Mean similarity score of retrieved chunks (0.0 to 1.0).

    Returns:
        float: Calibrated confidence in range [0.0, 1.0].
    """
    raw_confidence = (0.6 * verifier_score) + (0.4 * avg_evidence_score)
    return round(max(0.0, min(1.0, raw_confidence)), 3)


def confidence_node(state: AgentState) -> AgentState:
    """LangGraph node execution function for the Confidence Agent.

    Args:
        state: Current AgentState.

    Returns:
        AgentState: Updated state with calculated confidence and loop status.
    """
    verifier_score = state.get("verifier_score", 0.5)
    avg_evidence_score = state.get("avg_evidence_score", 0.5)
    retry_count = state.get("retry_count", 0)
    trace = list(state.get("trace", []))

    confidence = calculate_composite_confidence(verifier_score, avg_evidence_score)
    threshold = config.confidence_threshold
    max_retries = config.max_retries

    should_retry = (confidence < threshold) and (retry_count < max_retries)
    decision = "TRIGGER_RETRY" if should_retry else "APPROVE_ANSWER"

    badge_tier = (
        "HIGH"
        if confidence >= 0.8
        else ("MEDIUM" if confidence >= 0.5 else "LOW")
    )

    logger.info(
        f"Confidence Agent: score={confidence} ({badge_tier}), decision={decision}, retries={retry_count}/{max_retries}"
    )

    trace_entry = {
        "agent": "Confidence Agent",
        "action": f"Calibrated composite confidence: {confidence * 100:.1f}% -> {decision}",
        "status": "COMPLETED",
        "details": {
            "confidence": confidence,
            "badge_tier": badge_tier,
            "threshold": threshold,
            "verifier_score": verifier_score,
            "avg_evidence_score": avg_evidence_score,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "decision": decision,
        },
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    trace.append(trace_entry)

    # If triggering retry, increment retry_count for the next retrieval attempt
    next_retry_count = retry_count + 1 if should_retry else retry_count

    return {
        "confidence": confidence,
        "retry_count": next_retry_count,
        "should_retry": should_retry,
        "trace": trace,
    }
