"""Embedding management module for Multi-Agent Adaptive RAG.

Provides local, zero-cost vector embedding generation utilizing
sentence-transformers (all-MiniLM-L6-v2) or local Ollama embeddings (nomic-embed-text).
"""

from typing import List, Optional
import numpy as np
import requests

from config import config, setup_logger

logger = setup_logger("embeddings")


class LocalEmbeddingManager:
    """Manages embedding generation with local sentence-transformers or Ollama."""

    def __init__(self, model_name: Optional[str] = None):
        """Initializes the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model. Defaults to config value.
        """
        self.model_name = model_name or config.embedding_model_name
        self._st_model = None
        self._dimension = 384  # Default dimension for all-MiniLM-L6-v2

    @property
    def model(self):
        """Lazy loader for the SentenceTransformer model."""
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    f"Loading local SentenceTransformer model: {self.model_name}"
                )
                self._st_model = SentenceTransformer(self.model_name)
                # Check vector dimension
                test_vec = self._st_model.encode(["test"])
                self._dimension = len(test_vec[0])
                logger.info(
                    f"SentenceTransformer loaded successfully. Dimension: {self._dimension}"
                )
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer load failed ({e}). Checking Ollama embedding fallback."
                )
                self._st_model = "fallback"
        return self._st_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List[List[float]]: Vector representations.
        """
        if not texts:
            return []

        # Try SentenceTransformer first
        if self.model != "fallback":
            try:
                embeddings = self.model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False
                )
                return embeddings.tolist()
            except Exception as e:
                logger.warning(
                    f"Error in SentenceTransformer encoding: {e}. Falling back to Ollama/Heuristic."
                )

        # Fallback to Ollama embedding endpoint
        ollama_vecs = self._embed_via_ollama(texts)
        if ollama_vecs is not None:
            return ollama_vecs

        # Last-resort fallback: deterministic normalized hash projection
        logger.warning(
            "Using deterministic local pseudo-embedding (offline fallback)."
        )
        return [self._deterministic_vector(t, self._dimension) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generates an embedding vector for a single query string.

        Args:
            text: Query string.

        Returns:
            List[float]: Vector representation.
        """
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * self._dimension

    def _embed_via_ollama(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Attempts to obtain embeddings via local Ollama API."""
        try:
            url = f"{config.ollama_base_url}/api/embeddings"
            results = []
            for text in texts:
                res = requests.post(
                    url,
                    json={
                        "model": config.ollama_embed_model,
                        "prompt": text,
                    },
                    timeout=2.0,
                )
                if res.status_code == 200:
                    results.append(res.json().get("embedding", []))
                else:
                    return None
            return results
        except Exception:
            return None

    @staticmethod
    def _deterministic_vector(text: str, dim: int = 384) -> List[float]:
        """Deterministic pseudo-random vector based on text content for graceful offline testing."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31 - 1))
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# Global singleton instance
embedding_manager = LocalEmbeddingManager()
