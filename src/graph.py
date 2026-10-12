"""LangGraph Workflow definition for Multi-Agent Adaptive RAG.

Assembles the cyclical multi-agent graph with dynamic complexity routing,
self-verification loops, and Standard RAG baseline branches.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from config import config, setup_logger
from src.state import AgentState
from src.agents.planner import planner_node
from src.agents.evidence import evidence_node, draft_answer_node, standard_rag_node
from src.agents.verifier import verifier_node
from src.agents.confidence import confidence_node

logger = setup_logger("graph")


def route_planner_edge(state: AgentState) -> str:
    """Conditional router function following the Planner Agent decision.

    Args:
        state: Current AgentState.

    Returns:
        str: Next node identifier ('standard_rag' or 'evidence').
    """
    route = state.get("route", "complex")
    if route == "simple":
        return "standard_rag"
    return "evidence"


def route_confidence_edge(state: AgentState) -> str:
    """Conditional router function following the Confidence Agent evaluation.

    Args:
        state: Current AgentState.

    Returns:
        str: Next destination node ('evidence' for re-retrieval or END).
    """
    confidence = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)
    should_retry = state.get("should_retry", False)

    if should_retry:
        logger.info(
            f"Confidence {confidence:.2f} < threshold. Triggering adaptive re-retrieval loop (retry #{retry_count})."
        )
        return "evidence"

    logger.info(
        f"Answer approved with confidence {confidence:.2f} (retries={retry_count}). Ending workflow."
    )
    return END


def build_rag_graph():
    """Constructs and compiles the complete LangGraph StateGraph.

    Returns:
        CompiledStateGraph: Compiled LangGraph executable.
    """
    builder = StateGraph(AgentState)

    # 1. Register all nodes
    builder.add_node("planner", planner_node)
    builder.add_node("standard_rag", standard_rag_node)
    builder.add_node("evidence", evidence_node)
    builder.add_node("draft_answer", draft_answer_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("confidence", confidence_node)

    # 2. Add entrypoint
    builder.add_edge(START, "planner")

    # 3. Dynamic routing based on Planner Agent
    builder.add_conditional_edges(
        "planner",
        route_planner_edge,
        {
            "standard_rag": "standard_rag",
            "evidence": "evidence",
        },
    )

    # 4. Standard RAG terminates after single-shot retrieval and generation
    builder.add_edge("standard_rag", END)

    # 5. Multi-Agent pipeline flow
    builder.add_edge("evidence", "draft_answer")
    builder.add_edge("draft_answer", "verifier")
    builder.add_edge("verifier", "confidence")

    # 6. Adaptive verification & retry loop
    builder.add_conditional_edges(
        "confidence",
        route_confidence_edge,
        {
            "evidence": "evidence",
            END: END,
        },
    )

    return builder.compile()


# Compile graph singleton
marag_graph = build_rag_graph()


def run_pipeline(
    question: str,
    force_route: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> AgentState:
    """Runs a query through the LangGraph Multi-Agent RAG workflow.

    Args:
        question: User query string.
        force_route: Optional override ('simple' or 'complex').
        ground_truth: Optional ground truth answer.

    Returns:
        AgentState: Final state containing answers, evidence, confidence, and trace.
    """
    initial_state: AgentState = {
        "question": question,
        "route": force_route or "",
        "evidence": [],
        "evidence_sources": [],
        "draft_answer": "",
        "verified": False,
        "verifier_score": 0.0,
        "avg_evidence_score": 0.0,
        "confidence": 0.0,
        "retry_count": 0,
        "trace": [],
        "ground_truth": ground_truth,
        "is_standard_rag": (force_route == "simple"),
    }

    final_state = marag_graph.invoke(initial_state)
    return final_state


def run_standard_rag(
    question: str, ground_truth: Optional[str] = None
) -> AgentState:
    """Convenience runner for Standard RAG baseline.

    Args:
        question: User query string.
        ground_truth: Optional ground truth answer.

    Returns:
        AgentState: Baseline execution state.
    """
    return run_pipeline(question, force_route="simple", ground_truth=ground_truth)
