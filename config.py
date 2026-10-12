"""Configuration module for the Multi-Agent Adaptive RAG (MA-RAG) System.

Loads environment variables from .env and exposes strongly-typed configuration
parameters across data ingestion, vector indexing, agent orchestration, and UI.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = DATA_DIR / "chroma"
DOCS_DIR: Path = BASE_DIR / "docs"

# Load environment variables
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration parameters."""

    # Ollama LLM Runtime
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # Embeddings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # ChromaDB
    chroma_persist_directory: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR)
    )
    chroma_collection_name: str = "hotpot_qa_chunks"

    # Dataset
    dataset_name: str = os.getenv("DATASET_NAME", "hotpot_qa")
    dataset_config: str = os.getenv("DATASET_CONFIG", "distractor")
    dataset_split: str = os.getenv("DATASET_SPLIT", "validation[:200]")
    max_hotpot_samples: int = int(os.getenv("MAX_HOTPOT_SAMPLES", "200"))

    # Retrieval & Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Agent Control & Loop Limits
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton instance
config: AppConfig = AppConfig()


def setup_logger(name: str = "ma_rag") -> logging.Logger:
    """Configures and returns a standardized logger instance.

    Args:
        name: Logger identifier.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
