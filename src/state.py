"""LangGraph State Definition for Multi-Agent Adaptive RAG.

Defines the TypedDict state schema shared across all nodes in the workflow.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class TraceEntry(TypedDict, total=False):
    """Represents a single step in the agent execution trace."""

    agent: str
    action: str
    status: str
    details: Dict[str, Any]
    timestamp: str


class AgentState(TypedDict, total=False):
    """Global state container passed between LangGraph nodes.

    Attributes:
        question: User query or benchmark question.
        route: Decision from Planner ('simple' -> Standard RAG, 'complex' -> Multi-Agent).
        evidence: Retrieved context chunks from vector store.
        evidence_sources: Metadata associated with evidence chunks (e.g., titles/doc ids).
        draft_answer: Candidate response generated from retrieved evidence.
        verified: Whether Verifier Agent affirmed the answer is backed by evidence.
        verifier_score: Confidence score returned by Verifier (0.0 to 1.0).
        avg_evidence_score: Similarity / relevance metric of retrieved evidence.
        confidence: Combined weighted confidence score (0.0 to 1.0).
        retry_count: Number of retrieval refinement loops executed so far.
        trace: Chronological log of agent actions, rationales, and metrics.
        ground_truth: Optional reference answer (used during evaluation).
        is_standard_rag: Flag indicating if standard baseline execution path was taken.
        error: Any caught runtime warning or connection issue message.
    """

    question: str
    route: str  # "simple" or "complex"
    evidence: List[str]
    evidence_sources: List[Dict[str, Any]]
    draft_answer: str
    verified: bool
    verifier_score: float
    avg_evidence_score: float
    confidence: float
    retry_count: int
    should_retry: bool
    trace: List[Dict[str, Any]]
    ground_truth: Optional[str]
    is_standard_rag: bool
    error: Optional[str]
