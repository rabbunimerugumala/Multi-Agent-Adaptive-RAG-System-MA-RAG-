# 🚀 How to Run the System

This guide walks you through setting up, configuring, and launching the **Multi-Agent Adaptive RAG (MA-RAG)** system locally from scratch.

---

## 1. Prerequisites

Before installing the application, verify that your development environment meets the following specifications:

- **Python**: Version `3.11` or higher (`3.11+` recommended).
  - Verify with: `python --version`
- **Ollama**: Installed and running on your local machine.
  - Download from: [ollama.com](https://ollama.com/)
  - Verify with: `ollama --version`
- **Hardware Recommendations**:
  - Minimum: 8 GB RAM (CPU-only inference).
  - Recommended: 16 GB RAM with NVIDIA GPU (CUDA acceleration) for faster `llama3.1:8b` tokens/sec.
- **Disk Space**: Approximately 5.5 GB of free space for the quantized Llama 3.1 (8B) model and embedding weights.

---

## 2. Step-by-Step Terminal Commands

Follow these exact steps in your terminal or command prompt:

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/multi-agent-adaptive-rag.git
```

### Step 2: Navigate to the Project Directory
```bash
cd multi-agent-adaptive-rag
```

### Step 3: Create a Dedicated Virtual Environment
```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment
- **On Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```
- **On Windows (PowerShell)**:
  ```powershell
  venv\Scripts\activate
  ```
- **On Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

### Step 5: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Pull the Primary LLM via Ollama
```bash
ollama pull llama3.1:8b
```

### Step 7: Pull the Ollama Embedding Model (Optional Alternative)
```bash
ollama pull nomic-embed-text
```

---

## 3. Running the App

Start the interactive Streamlit web application:

```bash
streamlit run app.py
```

Streamlit will launch local server services and open your browser automatically at:
```
http://localhost:8501
```

### Running the Test Suite
To verify agent logic, confidence weighting, and LangGraph workflow integrity:
```bash
python -m pytest tests/test_agents.py -v
```

---

## 4. Troubleshooting & FAQ

### Issue A: Ollama is Not Running or Connection Refused
- **Symptom**: Sidebar shows a yellow badge saying `🟡 Ollama Offline`.
- **Cause**: The Ollama background service is not running or port `11434` is inaccessible.
- **Resolution**:
  1. Open a new terminal and start the server manually:
     ```bash
     ollama serve
     ```
  2. In another window, check if the server is responding:
     ```bash
     curl http://localhost:11434/api/tags
     ```
  3. *(Note: Even if Ollama is not active, MA-RAG will not crash. It automatically switches to its internal deterministic fallback synthesizer so you can test the UI and multi-agent graph immediately!)*

### Issue B: Re-initializing ChromaDB Vector Store
- **Symptom**: You want to clear the vector index, re-embed chunks, or refresh the cache.
- **Resolution**:
  1. Delete the persistent cache folder:
     - On Windows PowerShell:
       ```powershell
       Remove-Item -Recurse -Force data\chroma
       ```
     - On Linux/macOS:
       ```bash
       rm -rf data/chroma
       ```
  2. Re-index HotpotQA chunks using the initialization script:
     ```bash
     python -c "from src.vector_store import vector_store; print(vector_store.initialize_if_empty())"
     ```
  3. The documents will be freshly chunked, embedded via `all-MiniLM-L6-v2`, and persisted into `./data/chroma`.

### Issue C: HuggingFace Hub Rate Limits or Download Warning
- **Symptom**: Warning regarding unauthenticated HuggingFace requests or symlinks on Windows.
- **Resolution**:
  The application includes a bundled fallback dataset of curated HotpotQA multi-hop records. If HuggingFace Hub is unreachable or firewalled, the system automatically loads the fallback records without interruption.
