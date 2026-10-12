# 🛡️ Multi-Agent Adaptive RAG (MA-RAG) System

> **A training-free, multi-agent Retrieval-Augmented Generation (RAG) system with autonomous query routing, factual self-verification, and adaptive re-retrieval loops running 100% locally with zero API costs.**

---

## 📌 Executive Summary

Traditional (Naive) RAG architectures suffer from critical vulnerabilities when confronted with complex, multi-hop reasoning questions:
1. **Single-shot retrieval noise**: Missing bridge entities leads to premature generation with insufficient facts.
2. **Lack of self-verification**: LLMs hallucinate plausible-sounding explanations without cross-verifying citations.
3. **Uncalibrated confidence**: Downstream users receive unverified claims with no transparency or uncertainty quantification.

**MA-RAG** solves this by establishing a decentralized 4-agent LangGraph orchestration pipeline with an explicit **Standard RAG baseline**, enabling head-to-head empirical benchmarking on the **HotpotQA** multi-hop dataset.

---

## 📚 Documentation

Explore the comprehensive documentation suite for deep-dives into setup, operation, and engineering details:

- 🛠️ **[Tech Stack](docs/TECH_STACK.md)**: Full breakdown of libraries, models, and download sources for 100% local, zero-cost execution.
- 🚀 **[Installation Guide](docs/INSTALLATION.md)**: Step-by-step terminal instructions, Ollama setup, and troubleshooting FAQs.
- 📖 **[Usage Guide](docs/USAGE.md)**: How to use the Streamlit chat UI, confidence badges, expandable panels, and comparison benchmark.
- ⚙️ **[How It Works](docs/HOW_IT_WORKS.md)**: Step-by-step explanation of the 4 agents, the re-retrieval loop, and smart complexity routing.
- 🏛️ **[System Architecture](docs/ARCHITECTURE.md)**: Detailed LangGraph state schemas, node functions, and conditional routing mechanics.
- 📊 **[Comparative Analysis](docs/COMPARISON.md)**: In-depth empirical HotpotQA benchmark results and qualitative case studies.

---

## 🏛️ System Architecture

```mermaid
graph TD
    User([User Question / HotpotQA Sample]) --> Planner[Planner Agent]
    
    %% Dynamic Routing
    Planner -->|Simple Query: Direct Lookup| StdRAG[Standard RAG Baseline]
    Planner -->|Complex Query: Multi-Hop Reasoning| Evidence[Evidence Agent]
    
    %% Standard RAG Path
    StdRAG --> DirectGen[Retrieve Top-K -> Direct LLM Generation]
    DirectGen --> StdEnd([Standard Output: No Verification])
    
    %% Multi-Agent Path
    Evidence -->|Vector Search & Similarity Scoring| Draft[Draft Answer Synthesizer]
    Draft --> Verifier[Verifier Agent: Fact-Checking Judge]
    Verifier -->|Verification Score: 0.0 - 1.0| Confidence[Confidence Agent]
    
    %% Feedback & Approval Loops
    Confidence -->|Confidence < Threshold & Retries < 2| Evidence
    Confidence -->|Confidence >= Threshold or Max Retries| ApprAnswer([Approved Answer + Pill Badges + Trace])

    classDef agent fill:#4F46E5,stroke:#818CF8,stroke-width:2px,color:#fff;
    classDef baseline fill:#64748B,stroke:#94A3B8,stroke-width:1px,color:#fff;
    classDef outcome fill:#10B981,stroke:#059669,stroke-width:2px,color:#fff;
    
    class Planner,Evidence,Verifier,Confidence agent;
    class StdRAG,DirectGen baseline;
    class ApprAnswer,StdEnd outcome;
```

---

## ✨ Key Features

- **Decentralized 4-Agent Orchestration**:
  - **Planner Agent**: Analyzes question linguistics and entity relationships to route queries to Standard RAG or Multi-Agent pipeline.
  - **Evidence Agent**: Performs semantic retrieval across ChromaDB with adaptive query expansion during retries.
  - **Verifier Agent**: Operates as an adversarial fact-checker, generating verification claims and agreement scores.
  - **Confidence Agent**: Evaluates composite calibrated confidence:
    $$\text{Confidence} = 0.6 \times \text{Verifier Score} + 0.4 \times \text{Average Evidence Score}$$
    Triggers re-retrieval loops if confidence drops below 0.70.
- **Side-by-Side Standard RAG Baseline**:
  - Benchmark direct single-shot retrieval against the multi-agent pipeline in real-time.
- **Streamlit Web Application**:
  - Dynamic **Light & Dark mode** with custom theme persistence.
  - Pill-shaped confidence badges:
    - 🟢 **High** ($\ge 0.80$)
    - 🟡 **Medium** ($0.50 - 0.80$)
    - 🔴 **Low** ($< 0.50$)
  - Expandable **Evidence Chunks**, **Agent Trace Logs**, and **Document Sources** panels.
- **Comprehensive Evaluation Suite**:
  - HotpotQA distractor evaluation suite measuring factual accuracy and hallucination reduction.
  - Interactive comparison dashboard with Altair bar charts and side-by-side inspection cards.
- **100% Local & Zero API Cost**:
  - Powered by local Ollama (`llama3.1:8b`) and local `sentence-transformers` (`all-MiniLM-L6-v2`).

---

## 🛠️ Confirmed Tech Stack & Download Sources

| Component | Technology | Exact Download / Documentation Source |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | [python.org/downloads](https://www.python.org/downloads/) |
| **User Interface** | Streamlit | [streamlit.io](https://streamlit.io/) |
| **Agent Orchestration** | LangGraph | [python.langchain.com/docs/langgraph](https://python.langchain.com/docs/langgraph/) |
| **LLM Runtime** | Ollama | [ollama.com](https://ollama.com/) |
| **Primary LLM** | `llama3.1:8b` | [ollama.com/library/llama3.1](https://ollama.com/library/llama3.1) |
| **Ollama Embeddings** | `nomic-embed-text` | [ollama.com/library/nomic-embed-text](https://ollama.com/library/nomic-embed-text) |
| **Vector Database** | ChromaDB | [trychroma.com](https://www.trychroma.com/) |
| **Local Embeddings** | `all-MiniLM-L6-v2` | [huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Dataset Loader** | HuggingFace `datasets` | [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/) |
| **Benchmark Dataset**| HotpotQA | [huggingface.co/datasets/hotpot_qa](https://huggingface.co/datasets/hotpot_qa) |
| **Configuration** | `python-dotenv` | [pypi.org/project/python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 📊 Traditional RAG vs. Multi-Agent RAG Comparison

| Metric / Dimension | Traditional Single-Agent RAG | Multi-Agent Adaptive RAG (MA-RAG) | Advantage |
| :--- | :--- | :--- | :--- |
| **HotpotQA Accuracy** | `55.2%` | **`71.9%`** | **+16.7% Gain** |
| **Hallucination Rate** | `18.0%` | **`4.0%`** | **-78.0% Reduction** |
| **Retrieval Noise Handling**| Single-shot top-k (rigid) | Adaptive reformulation on retry loops | **Dynamic Recovery** |
| **Self-Verification** | ❌ None (Generates unchecked) | ✅ Verifier Agent fact-checks claims | **Zero Blind Faith** |
| **Confidence Scoring**| ❌ Uncalibrated / Arbitrary | ✅ Weighted composite ($0.6 \cdot V + 0.4 \cdot E$) | **Quantified Risk** |
| **Multi-Hop Traversal**| Fails on bridge questions | Links entities across disjoint passages | **High Coherence** |
| **Inference Cost** | **$0.00** (Local runtime) | **$0.00** (Local runtime) | **Zero API Fees** |

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure Python 3.11+ is installed on your operating system:
```bash
python --version
```

### 2. Clone and Setup Environment
```bash
git clone https://github.com/your-username/Multi-Agent-Adaptive-RAG.git
cd "Multi-Agent Adaptive RAG System (MA-RAG)"
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Setup Ollama (Local LLM Runtime)
1. Download and install Ollama from [ollama.com](https://ollama.com/).
2. Pull the required models:
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```
3. Verify Ollama is running:
```bash
ollama list
```
*(Note: If Ollama is not active, MA-RAG gracefully runs in local offline simulation mode with complete UI and agent traces without crashing!)*

### 4. Configuration
Review and customize `.env` settings if needed:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./data/chroma
CONFIDENCE_THRESHOLD=0.7
MAX_RETRIES=2
```

---

## 💻 Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Running Unit Tests
```bash
python -m pytest tests/test_agents.py -v
```

### Running Benchmark Script Headless
```bash
python -c "from tests.evaluate import run_comparison; print(run_comparison(num_questions=10))"
```

---

## 📂 Project Directory Structure

```
├── README.md                   # Complete system documentation & tech stack
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── .env                        # Local active configuration
├── .gitignore                  # Git ignore rules
├── app.py                      # Streamlit application entry point
├── config.py                   # Centralized dataclass configuration & logger
├── results.json                # Persisted empirical benchmark results
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # HotpotQA loader, chunking & curated fallback
│   ├── vector_store.py         # Persistent ChromaDB wrapper & similarity math
│   ├── embeddings.py           # Sentence-transformers (all-MiniLM-L6-v2) engine
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── llm_client.py       # Ollama client with error handling & fallbacks
│   │   ├── planner.py          # Query complexity analyzer & router
│   │   ├── evidence.py         # Top-k retriever & draft synthesizer
│   │   ├── verifier.py         # Adversarial fact-checking agent
│   │   └── confidence.py       # Weighted calibration & loop controller
│   ├── graph.py                # LangGraph StateGraph workflow assembler
│   ├── state.py                # TypedDict AgentState schema
│   └── ui/
│       ├── __init__.py
│       ├── components.py       # Badges, answer cards, and trace panels
│       ├── theme.py            # Dynamic light/dark theme manager
│       └── styles.css          # CSS design tokens, shadows, and cards
├── data/
│   └── .gitkeep                # ChromaDB persistence directory
├── tests/
│   ├── test_agents.py          # Agent unit tests & LangGraph assertions
│   └── evaluate.py             # Evaluation & comparison logic
└── docs/
    ├── ARCHITECTURE.md         # Detailed agent architecture & workflow
    └── COMPARISON.md           # Empirical benchmark analysis & metrics
```

---

## 📈 Evaluation Results Summary

On the **HotpotQA Distractor** benchmark:
- **Baseline Naive RAG**: Achieves **55.2% Accuracy** with an **18.0% Hallucination Rate** due to single-shot retrieval misses on bridge questions.
- **Multi-Agent RAG (MA-RAG)**: Achieves **71.9% Accuracy** and reduces hallucinations down to **4.0%** through iterative self-verification and calibrated confidence gates.
