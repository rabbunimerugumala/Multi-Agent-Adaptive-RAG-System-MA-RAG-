"""Planner Agent module for Multi-Agent Adaptive RAG.

Analyzes the input question complexity, detects multi-hop reasoning requirements,
and routes questions to either Standard RAG (simple) or the full 4-agent pipeline (complex).
"""

from datetime import datetime
import re
from typing import Dict, Any

from config import setup_logger
from src.state import AgentState
from src.agents.llm_client import llm_client

logger = setup_logger("planner_agent")


def analyze_complexity(question: str) -> Dict[str, Any]:
    """Evaluates question complexity using LLM analysis and structural heuristics.

    Args:
        question: User query or HotpotQA question.

    Returns:
        Dict[str, Any]: Route decision ('simple' or 'complex') with rationale.
    """
    prompt = f"""You are the Planner Agent in an Adaptive RAG system.
Your job is to decide whether the question requires:
1. "SIMPLE": A single-step direct lookup or definition.
2. "COMPLEX": Multi-hop reasoning, comparing multiple entities, finding bridge relationships, or resolving dependencies between documents.

Question: {question}

Format your output strictly as:
ROUTE: [SIMPLE or COMPLEX]
REASONING: [Brief explanation of the question's reasoning hops]
"""
    response = llm_client.generate(prompt, temperature=0.0, max_tokens=150)

    route = "complex"  # Default to complex for HotpotQA benchmark safety
    reasoning = "Multi-hop query detected."

    match = re.search(r"ROUTE:\s*(SIMPLE|COMPLEX)", response, re.IGNORECASE)
    if match:
        route = match.group(1).lower()

    reason_match = re.search(r"REASONING:\s*(.+)", response, re.IGNORECASE)
    if reason_match:
        reasoning = reason_match.group(1).strip()
    else:
        reasoning = response.strip()[:150]

    logger.info(f"Planner decision for '{question[:40]}...': route={route}")
    return {"route": route, "reasoning": reasoning}


def planner_node(state: AgentState) -> AgentState:
    """LangGraph node execution function for the Planner Agent.

    Args:
        state: Current AgentState.

    Returns:
        AgentState: Updated state with route decision and trace log.
    """
    question = state.get("question", "")
    trace = list(state.get("trace", []))

    # Allow user override if pre-set in state, otherwise evaluate
    route = state.get("route")
    reasoning = "Route pre-configured."

    if not route or route not in ["simple", "complex"]:
        plan_result = analyze_complexity(question)
        route = plan_result["route"]
        reasoning = plan_result["reasoning"]

    trace_entry = {
        "agent": "Planner Agent",
        "action": f"Classified question complexity as '{route.upper()}'",
        "status": "COMPLETED",
        "details": {
            "route": route,
            "reasoning": reasoning,
            "target_pipeline": "Standard RAG" if route == "simple" else "Multi-Agent Verifier Pipeline",
        },
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    trace.append(trace_entry)

    return {
        "route": route,
        "trace": trace,
    }
