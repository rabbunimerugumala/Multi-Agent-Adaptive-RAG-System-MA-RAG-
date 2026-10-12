# 🛠️ Technology Stack & Sources

This document details every core library, framework, runtime engine, dataset, and model powering the **Multi-Agent Adaptive RAG (MA-RAG)** system.

---

## 📋 Comprehensive Tech Stack Matrix

| Layer | Technology | Version | Purpose | Official Download/Source Link |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | Python | `3.11+` | Core programming language runtime for agent logic, data loaders, and graph execution. | [python.org/downloads](https://www.python.org/downloads/) |
| **User Interface** | Streamlit | `1.35.0+` | Interactive web dashboard, dynamic light/dark theme manager, live charts, and agent trace viewer. | [streamlit.io](https://streamlit.io/) |
| **Orchestration** | LangGraph | `0.2.0+` | Cyclic state graph orchestrator managing multi-agent state transitions, conditional edges, and re-retrieval loops. | [python.langchain.com/docs/langgraph](https://python.langchain.com/docs/langgraph/) |
| **LLM Runtime** | Ollama | `0.3.0+` | High-performance local LLM runtime for running quantized models on local GPU or CPU. | [ollama.com](https://ollama.com/) |
| **LLM Model** | `llama3.1:8b` | `8B (Q4_K_M)` | Foundational instruction-tuned language model used for planning, draft generation, and adversarial claim verification. | [ollama.com/library/llama3.1](https://ollama.com/library/llama3.1) |
| **Embeddings (Ollama)** | `nomic-embed-text` | `v1.5` | High-dimensional dense text embedding model running inside Ollama runtime. | [ollama.com/library/nomic-embed-text](https://ollama.com/library/nomic-embed-text) |
| **Embeddings (Local CPU)** | `all-MiniLM-L6-v2` | `Sentence-Transformers 3.0+` | Ultra-fast local dense bi-encoder generating 384-dimensional semantic embeddings on local CPU. | [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB | `0.5.0+` | Embedded persistent vector database storing chunk embeddings, document metadata, and computing cosine distances. | [trychroma.com](https://www.trychroma.com/) |
| **Benchmark Dataset** | HotpotQA | `distractor` split | Benchmark dataset featuring complex, multi-hop questions requiring cross-document entity synthesis. | [huggingface.co/datasets/hotpot_qa](https://huggingface.co/datasets/hotpot_qa) |
| **Dataset Loader** | HuggingFace Datasets | `2.19.0+` | Automated loader and stream pipeline for downloading and caching HotpotQA validation partitions. | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/) |
| **Configuration** | `python-dotenv` | `1.0.0+` | Manages environment configurations, model parameters, chunk dimensions, and calibration thresholds. | [pypi.org/project/python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 💸 Why This Stack is 100% Free and 100% Local

A cornerstone design requirement of **MA-RAG** is total computational sovereignty and zero operational expenditure ($0.00 API costs):

1. **No External API Dependencies**: Unlike commercial RAG stacks that incur per-token charges with OpenAI, Anthropic, or Cohere APIs, MA-RAG executes inference entirely on your local machine using **Ollama** (`llama3.1:8b`).
2. **Local CPU Vector Embeddings**: Document vectorization uses local **Sentence-Transformers** (`all-MiniLM-L6-v2`) or **Ollama** (`nomic-embed-text`). These operate with zero cloud egress and require no paid embedding services.
3. **Embedded Persistent Vector DB**: **ChromaDB** runs locally as an embedded SQLite and Parquet-backed vector store in `./data/chroma`, removing any need for cloud-hosted vector SaaS providers like Pinecone or Weaviate.
4. **Offline Resilience**: The system features integrated local fallbacks and cached HotpotQA distractor partitions, allowing full testing, UI interactions, and graph evaluations even without an active internet connection.
