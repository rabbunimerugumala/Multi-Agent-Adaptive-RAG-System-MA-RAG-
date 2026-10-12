"""Verifier Agent module for Multi-Agent Adaptive RAG.

Evaluates factual consistency and evidence grounding for the draft answer,
returning a verification flag and confidence score.
"""

from datetime import datetime
from typing import Dict, Any

from config import setup_logger
from src.state import AgentState
from src.agents.llm_client import llm_client

logger = setup_logger("verifier_agent")


def verify_answer_against_evidence(
    question: str, evidence: list, draft_answer: str
) -> Dict[str, Any]:
    """Invokes LLM verification to detect hallucinations or ungrounded claims.

    Args:
        question: User query.
        evidence: Retrieved context chunks.
        draft_answer: Candidate answer to audit.

    Returns:
        Dict[str, Any]: Verification outcome containing verified (bool), score (float), rationale.
    """
    evidence_text = "\n\n".join(
        [f"[Doc {i+1}] {t}" for i, t in enumerate(evidence)]
    )

    return llm_client.verify_answer(
        question=question,
        evidence=evidence_text,
        draft_answer=draft_answer,
    )


def verifier_node(state: AgentState) -> AgentState:
    """LangGraph node execution function for the Verifier Agent.

    Args:
        state: Current AgentState.

    Returns:
        AgentState: Updated state with verification results and trace.
    """
    question = state.get("question", "")
    evidence = state.get("evidence", [])
    draft_answer = state.get("draft_answer", "")
    trace = list(state.get("trace", []))

    result = verify_answer_against_evidence(question, evidence, draft_answer)
    verified = result.get("verified", False)
    score = result.get("score", 0.5)
    rationale = result.get("rationale", "")

    status_icon = "PASS" if verified else "FAIL"
    logger.info(f"Verifier Agent outcome: {status_icon} (score={score})")

    trace_entry = {
        "agent": "Verifier Agent",
        "action": f"Fact-checked draft answer against retrieved evidence: [{status_icon}]",
        "status": "COMPLETED",
        "details": {
            "verified": verified,
            "verifier_score": score,
            "rationale": rationale,
            "evidence_evaluated_chunks": len(evidence),
        },
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    trace.append(trace_entry)

    return {
        "verified": verified,
        "verifier_score": score,
        "trace": trace,
    }
