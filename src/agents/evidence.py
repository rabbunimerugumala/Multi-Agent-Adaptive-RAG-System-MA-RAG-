"""Evidence Agent module for Multi-Agent Adaptive RAG.

Queries ChromaDB vector store, performs adaptive query reformulation on retries,
and handles draft answer generation as well as baseline Standard RAG execution.
"""

from datetime import datetime
from typing import List, Dict, Any

from config import config, setup_logger
from src.state import AgentState
from src.vector_store import vector_store
from src.agents.llm_client import llm_client

logger = setup_logger("evidence_agent")


def retrieve_evidence(
    query: str, top_k: int = 5, retry_count: int = 0
) -> List[Dict[str, Any]]:
    """Retrieves top-k evidence chunks from ChromaDB with adaptive retry query expansion.

    Args:
        query: User question or search query.
        top_k: Number of chunks to retrieve.
        retry_count: Current retry iteration for adaptive reformulation.

    Returns:
        List[Dict[str, Any]]: Retrieved chunks with text, title, and similarity score.
    """
    effective_query = query
    if retry_count > 0:
        # Adaptive query refinement on retry loops
        # Expand query with focal entities or sub-questions
        prompt = f"""Given this question that previously yielded insufficient evidence:
Question: {query}
Generate an alternative, highly focused search query focusing on specific named entities or supporting facts.
Alternative query:"""
        refined = llm_client.generate(prompt, temperature=0.2, max_tokens=60)
        refined_clean = refined.strip().strip('"').strip("'")
        if refined_clean and len(refined_clean) > 5:
            effective_query = f"{query} {refined_clean}"
            logger.info(f"Adaptive reformulation (retry {retry_count}): '{effective_query}'")

    results = vector_store.query(effective_query, top_k=top_k)
    return results


def evidence_node(state: AgentState) -> AgentState:
    """LangGraph node execution function for the Evidence Agent.

    Args:
        state: Current AgentState.

    Returns:
        AgentState: Updated state with retrieved evidence chunks and trace.
    """
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)
    trace = list(state.get("trace", []))

    top_k = config.top_k
    chunks = retrieve_evidence(question, top_k=top_k, retry_count=retry_count)

    evidence_texts = [c["text"] for c in chunks]
    evidence_sources = [
        {"title": c["title"], "score": c["score"], "metadata": c["metadata"]}
        for c in chunks
    ]

    scores = [c["score"] for c in chunks] if chunks else [0.0]
    avg_evidence_score = round(sum(scores) / max(1, len(scores)), 3)

    action_label = (
        f"Retrieved top-{len(chunks)} chunks (Initial Search)"
        if retry_count == 0
        else f"Re-retrieved top-{len(chunks)} chunks (Adaptive Loop #{retry_count})"
    )

    trace_entry = {
        "agent": "Evidence Agent",
        "action": action_label,
        "status": "COMPLETED",
        "details": {
            "chunks_found": len(chunks),
            "avg_similarity_score": avg_evidence_score,
            "sources": [c["title"] for c in chunks],
            "retry_iteration": retry_count,
        },
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    trace.append(trace_entry)

    return {
        "evidence": evidence_texts,
        "evidence_sources": evidence_sources,
        "avg_evidence_score": avg_evidence_score,
        "trace": trace,
    }


def draft_answer_node(state: AgentState) -> AgentState:
    """Generates an initial draft answer synthesized strictly from retrieved evidence.

    Args:
        state: Current AgentState.

    Returns:
        AgentState: Updated state with draft_answer and trace.
    """
    question = state.get("question", "")
    evidence = state.get("evidence", [])
    trace = list(state.get("trace", []))

    evidence_block = "\n\n".join(
        [f"[Doc {i+1}] {text}" for i, text in enumerate(evidence)]
    )

    prompt = f"""You are a precise question-answering assistant.
Answer the following question based ONLY on the provided evidence.
If comparing entities, clearly state the factual comparison.
Be concise, clear, and direct.

Question: {question}

Evidence:
{evidence_block}

Answer:"""

    draft_answer = llm_client.generate(prompt, temperature=0.0, max_tokens=256)
    if not draft_answer.strip():
        draft_answer = "Unable to deduce answer from the available documents."

    trace_entry = {
        "agent": "Evidence Agent",
        "action": "Synthesized candidate draft answer from evidence",
        "status": "COMPLETED",
        "details": {
            "draft_answer_preview": draft_answer[:120] + ("..." if len(draft_answer) > 120 else ""),
            "evidence_chunk_count": len(evidence),
        },
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    trace.append(trace_entry)

    return {
        "draft_answer": draft_answer,
        "trace": trace,
    }


def standard_rag_node(state: AgentState) -> AgentState:
    """Executes the Naive/Standard RAG baseline pipeline (no verification, no retry loops).

    Args:
        state: Current AgentState.

    Returns:
        AgentState: Final state for Standard RAG path.
    """
    question = state.get("question", "")
    trace = list(state.get("trace", []))

    # 1. Retrieve top-k chunks directly
    chunks = retrieve_evidence(question, top_k=config.top_k, retry_count=0)
    evidence_texts = [c["text"] for c in chunks]
    evidence_sources = [
        {"title": c["title"], "score": c["score"], "metadata": c["metadata"]}
        for c in chunks
    ]

    scores = [c["score"] for c in chunks] if chunks else [0.0]
    avg_evidence_score = round(sum(scores) / max(1, len(scores)), 3)

    evidence_block = "\n\n".join(
        [f"[Doc {i+1}] {text}" for i, text in enumerate(evidence_texts)]
    )

    prompt = f"""Answer the question based on the provided context.
Question: {question}

Context:
{evidence_block}

Answer:"""

    answer = llm_client.generate(prompt, temperature=0.2, max_tokens=256)

    trace_entry = {
        "agent": "Standard RAG Baseline",
        "action": "Single-shot Retrieve & Generate (No self-verification, no adaptive retry)",
        "status": "COMPLETED",
        "details": {
            "chunks_retrieved": len(chunks),
            "avg_similarity": avg_evidence_score,
            "pipeline": "Standard RAG (Direct Retrieve -> Generate)",
        },
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    }
    trace.append(trace_entry)

    return {
        "evidence": evidence_texts,
        "evidence_sources": evidence_sources,
        "draft_answer": answer,
        "verified": False,
        "verifier_score": 0.5,
        "avg_evidence_score": avg_evidence_score,
        "confidence": 0.5,  # Uncalibrated baseline
        "is_standard_rag": True,
        "trace": trace,
    }
