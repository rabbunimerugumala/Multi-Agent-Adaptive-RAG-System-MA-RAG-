"""ChromaDB Vector Store wrapper for Multi-Agent Adaptive RAG.

Handles persistent indexing, similarity search, caching, and distance-to-similarity
score normalization for HotpotQA chunks.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config import config, setup_logger
from src.embeddings import embedding_manager
from src.data_loader import DocumentChunk, HotpotQADataLoader

logger = setup_logger("vector_store")


class VectorStoreManager:
    """Encapsulates persistent ChromaDB operations and vector similarity queries."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Initializes the ChromaDB client and collection.

        Args:
            persist_directory: Local disk path for Chroma persistence.
            collection_name: Name of the collection.
        """
        self.persist_directory = Path(
            persist_directory or config.chroma_persist_directory
        )
        self.collection_name = collection_name or config.chroma_collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collection = None
        self._fallback_docs: List[DocumentChunk] = []
        self._fallback_vectors: Optional[np.ndarray] = None

    def _init_chroma(self):
        """Lazy initialization of Chroma client and collection."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            logger.info(f"Connecting to ChromaDB at: {self.persist_directory}")
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Collection '{self.collection_name}' ready. Current count: {self._collection.count()}"
            )
        except Exception as e:
            logger.warning(
                f"ChromaDB initialization failed ({e}). Operating in memory-backed mode."
            )
            self._collection = None

    def initialize_if_empty(self, data_loader: Optional[HotpotQADataLoader] = None) -> int:
        """Indexes HotpotQA chunks if the collection is empty.

        Cached to never re-embed on every startup.

        Args:
            data_loader: DataLoader instance.

        Returns:
            int: Number of total documents indexed.
        """
        self._init_chroma()

        if self._collection is not None:
            count = self._collection.count()
            if count > 0:
                logger.info(f"ChromaDB cache hit! Found {count} existing indexed documents.")
                return count

        logger.info("Vector store is empty. Ingesting and embedding HotpotQA documents...")
        loader = data_loader or HotpotQADataLoader()
        samples = loader.load_dataset(config.max_hotpot_samples)
        chunks = loader.prepare_chunks(samples)

        if not chunks:
            logger.warning("No chunks generated for vector store.")
            return 0

        self.add_documents(chunks)
        return len(chunks)

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Embeds and persists document chunks in ChromaDB.

        Args:
            chunks: List of DocumentChunk instances.
        """
        self._init_chroma()

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Compute embeddings
        logger.info(f"Embedding {len(chunks)} text chunks...")
        embeddings = embedding_manager.embed_documents(texts)

        if self._collection is not None:
            # Add in batches of 100 to avoid message size limits
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_i = min(i + batch_size, len(chunks))
                self._collection.add(
                    ids=ids[i:end_i],
                    documents=texts[i:end_i],
                    embeddings=embeddings[i:end_i],
                    metadatas=metadatas[i:end_i],
                )
            logger.info(f"Successfully saved {len(chunks)} chunks to ChromaDB.")
        else:
            # Fallback in-memory storage
            self._fallback_docs = chunks
            self._fallback_vectors = np.array(embeddings, dtype=np.float32)
            logger.info(f"Saved {len(chunks)} chunks to in-memory fallback index.")

    def query(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Queries the vector index for top-k nearest document chunks.

        Args:
            query_text: Search query.
            top_k: Number of chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved chunks with text, title, and similarity score (0.0 to 1.0).
        """
        self._init_chroma()
        query_vec = embedding_manager.embed_query(query_text)

        if self._collection is not None and self._collection.count() > 0:
            try:
                results = self._collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(top_k, self._collection.count()),
                    include=["documents", "metadatas", "distances"],
                )

                chunks_out: List[Dict[str, Any]] = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                dists = results.get("distances", [[]])[0]

                for doc, meta, dist in zip(docs, metas, dists):
                    # For cosine distance, similarity = 1.0 - (dist / 2.0) or max(0, 1.0 - dist)
                    # Chroma cosine distance is in [0, 2]
                    similarity = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                    chunks_out.append(
                        {
                            "text": doc,
                            "title": meta.get("title", "Unknown"),
                            "score": round(similarity, 4),
                            "metadata": meta,
                        }
                    )
                return chunks_out
            except Exception as e:
                logger.warning(f"Error querying ChromaDB: {e}. Falling back to in-memory.")

        # In-memory vector search fallback
        if self._fallback_vectors is not None and len(self._fallback_docs) > 0:
            q_norm = np.array(query_vec, dtype=np.float32)
            q_norm_val = np.linalg.norm(q_norm)
            if q_norm_val > 0:
                q_norm /= q_norm_val

            sims = np.dot(self._fallback_vectors, q_norm)
            top_indices = np.argsort(sims)[::-1][:top_k]

            chunks_out = []
            for idx in top_indices:
                doc = self._fallback_docs[idx]
                score = max(0.0, min(1.0, float(sims[idx])))
                chunks_out.append(
                    {
                        "text": doc.text,
                        "title": doc.doc_title,
                        "score": round(score, 4),
                        "metadata": doc.metadata,
                    }
                )
            return chunks_out

        return []


# Global singleton instance
vector_store = VectorStoreManager()
