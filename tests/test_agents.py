"""Unit tests for the Multi-Agent Adaptive RAG System.

Validates Planner, Evidence, Verifier, Confidence agents, and LangGraph workflow.
"""

import pytest
from src.state import AgentState
from src.data_loader import chunk_text
from src.agents.confidence import calculate_composite_confidence, confidence_node
from src.agents.planner import analyze_complexity, planner_node
from src.graph import marag_graph, run_pipeline, run_standard_rag


def test_chunk_text():
    """Validates word/token based text chunking and overlap."""
    sample_text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_text(sample_text, chunk_size=30, overlap=10)

    assert len(chunks) > 1
    assert "word0" in chunks[0]
    # Check overlap
    assert "word25" in chunks[0]
    assert "word25" in chunks[1]


def test_confidence_calculation():
    """Validates the mathematical formula: 0.6 * verifier + 0.4 * evidence."""
    # Perfect score: 0.6(1.0) + 0.4(1.0) = 1.0
    assert calculate_composite_confidence(1.0, 1.0) == 1.0

    # Minimum score: 0.6(0.0) + 0.4(0.0) = 0.0
    assert calculate_composite_confidence(0.0, 0.0) == 0.0

    # Balanced score: 0.6(0.9) + 0.4(0.7) = 0.54 + 0.28 = 0.82
    assert calculate_composite_confidence(0.9, 0.7) == 0.82

    # Borderline threshold check (0.70)
    # 0.6(0.8) + 0.4(0.55) = 0.48 + 0.22 = 0.70
    assert calculate_composite_confidence(0.8, 0.55) == 0.70


def test_confidence_node_retry_logic():
    """Validates retry count incrementing and state mutation."""
    # Case 1: Low confidence triggers retry
    low_state: AgentState = {
        "verifier_score": 0.4,
        "avg_evidence_score": 0.5,
        "retry_count": 0,
        "trace": [],
    }
    result = confidence_node(low_state)
    assert result["confidence"] < 0.7
    assert result["retry_count"] == 1
    assert len(result["trace"]) == 1

    # Case 2: High confidence approves without retry increment
    high_state: AgentState = {
        "verifier_score": 0.95,
        "avg_evidence_score": 0.90,
        "retry_count": 0,
        "trace": [],
    }
    result_high = confidence_node(high_state)
    assert result_high["confidence"] >= 0.7
    assert result_high["retry_count"] == 0


def test_planner_agent():
    """Validates question complexity analysis."""
    # Complex multi-hop query
    res = analyze_complexity(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    )
    assert res["route"] in ["simple", "complex"]

    # Test planner_node
    state: AgentState = {
        "question": "Which band has more members: Radiohead or The Police?",
        "trace": [],
    }
    node_out = planner_node(state)
    assert "route" in node_out
    assert len(node_out["trace"]) == 1


def test_standard_rag_pipeline_execution():
    """Validates execution of the baseline Standard RAG pipeline."""
    question = "Who directed Doctor Strange?"
    final_state = run_standard_rag(question)

    assert "draft_answer" in final_state
    assert len(final_state["draft_answer"]) > 0
    assert final_state.get("is_standard_rag") is True
    assert len(final_state.get("trace", [])) >= 1


def test_marag_graph_structure():
    """Validates that LangGraph compiles and has required node keys."""
    assert marag_graph is not None
    # Test workflow invocation
    state = run_pipeline(
        question="What magazine did David Aukin co-found?",
        force_route="complex",
    )
    assert "draft_answer" in state
    assert "confidence" in state
    assert "verified" in state
    assert len(state.get("trace", [])) >= 3
