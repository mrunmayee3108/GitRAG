# GitRAG ⚡

A high-performance codebase intelligence and Retrieval-Augmented Generation (RAG) engine designed to index, search, and understand software repositories with symbol-level precision, exact line-number citations, and interactive architectural flow diagrams.

---

## 🌟 Overview

**GitRAG** transforms source code repositories and documentation into an AI-ready semantic vector database and interactive intelligence dashboard. Unlike generic text chunkers that blindly slice code and lose structural context, GitRAG uses **AST (Abstract Syntax Tree)** parsing to extract meaningful semantic symbols (classes, functions, methods, docstrings, and imports) while maintaining exact 1-indexed line references.

Retrieved code chunks are enriched with contextual metadata headers, embedded using dense vector representations via **FAISS**, and synthesized by **Google's Gemini** (`google-genai` SDK) to answer technical questions with exact inline citations and interactive **Mermaid.js** execution flow diagrams.

---

## ✨ Features

- 🌳 **AST-Powered Symbol Extraction**: Parses Python code into standalone functions, classes, and methods using Python's native `ast` module.
- 📑 **Multi-Language & Docs Support**: Supports Python (`.py`), JavaScript/TypeScript (`.js`, `.ts`, `.jsx`, `.tsx`), and Markdown (`.md`).
- 🛡️ **Zip-Slip Safe Archive Extraction**: Safely inspects and extracts uploaded `.zip` repositories while preventing path traversal vulnerabilities (`..` escape attacks).
- 🧩 **Context-Aware Semantic Chunking**: Enriches each chunk with a standardized header (`File: ... (Lines start-end) | Type: ... | Name: ...`) and handles oversized symbols with intelligent sliding-window overlap.
- ⚡ **FAISS Vector Search**: Builds L2-normalized Inner Product indices (equivalent to Cosine Similarity) for fast, dense semantic retrieval.
- 🤖 **Grounded Repository QA (Gemini)**: Answers developer questions with strict grounding on retrieved code snippets, preventing hallucinations.
- 🏷️ **Exact Line-Number Citations**: Automatically cites references inline in the format `[file_path:start-end]` (e.g., `[indexer/parser.py:45-80]`).
- 🗺️ **Code Flow & Execution Visualizer**: Automatically maps caller-callee sequences and architectural dependencies into interactive **Mermaid.js** flowcharts.
- 🖥️ **Streamlit UI Dashboard**: Complete 3-tab developer workspace:
  - **💬 Code Q&A Chat**: Conversational interface with interactive citation badges and expandable code inspector.
  - **🗺️ Code Flow Visualizer**: Live rendered Mermaid diagrams with pan/zoom and copyable definitions.
  - **📂 Repository Structure**: Collapsible file hierarchy and symbol browser with full code inspection.
- 🔐 **Secure Server-Side Configuration**: `GEMINI_API_KEY` is loaded exclusively from the server environment or `.env` file—never exposed in the UI or committed to Git.
- 🧪 **Self-Contained Test Suites**: Comprehensive end-to-end integration tests (`test_indexer.py` and `test_rag.py`).

---

## 📁 Repository Structure

```text
GitRAG/
├── indexer/
│   ├── __init__.py          # Public package exports
│   ├── parser.py            # AST parsing, file scanning, and safe ZIP extraction
│   ├── chunker.py           # Contextual chunking with metadata headers
│   └── vector_store.py      # FAISS vector store & SentenceTransformer retrieval
├── rag/
│   ├── __init__.py          # Citation RAG and code flow exports
│   ├── citation_rag.py      # Gemini grounded Q&A with exact line citations
│   └── code_flow.py         # Caller-callee & architectural Mermaid flow generator
├── app.py                   # Full interactive Streamlit Dashboard
├── requirements.txt         # Project dependencies
├── test_indexer.py          # Part 1 end-to-end integration test suite
├── test_rag.py              # Part 2 RAG & flow integration test suite
├── .env.example             # Environment variable template
├── .gitignore               # Security rules (ignores .env, .venv, cache)
└── README.md                # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ (Python 3.10 to 3.13 supported)
- `pip` or virtual environment manager (`venv` / `conda`)
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### 2. Installation

Clone this repository and create a virtual environment:

```bash
# Clone the repository
git clone https://github.com/mrunmayee3108/GitRAG.git
cd GitRAG

# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

*(Note: `.env` is ignored by Git and will never be committed).*

---

## 🖥️ Running the Streamlit Dashboard

Launch the interactive GitRAG dashboard:

```bash
streamlit run app.py
```

Open your browser to:
```text
http://localhost:8501
```

### Dashboard Capabilities:
1. **Sidebar**:
   - Select Gemini reasoning model (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`).
   - Select repository source: **Local Directory** or **Upload ZIP Archive**.
   - Adjust Top-K retrieved snippets.
   - Click **"🚀 Index Repository"** to parse AST symbols and build the FAISS index.
2. **💬 Tab 1 ("Code Q&A Chat")**:
   - Ask complex architectural and implementation questions.
   - Click on file/line badge citations.
   - Expand the **"Retrieved Code Snippets"** viewer to inspect raw code and similarity scores.
3. **🗺️ Tab 2 ("Code Flow Visualizer")**:
   - View caller-callee sequence diagrams and data flow rendered via Mermaid.js.
   - Enter custom flow queries (e.g., *"Trace how user authentication flows into token generation"*).
4. **📂 Tab 3 ("Repository Structure")**:
   - Browse the repository module tree.
   - Filter functions, classes, and methods with line counts and source preview.

---

## 💡 Python API Usage

### 1. Parse a Repository or ZIP Archive

```python
from pathlib import Path
from indexer import parse_repository

# Parse a directory
symbols = parse_repository(Path("./sample_repo"))

# Or parse a ZIP archive safely
symbols = parse_repository(Path("./sample_repo.zip"))

print(f"Extracted {len(symbols)} code symbols.")
```

### 2. Chunk Symbols with Contextual Headers

```python
from indexer import chunk_symbols

chunks = chunk_symbols(
    symbols,
    max_lines_per_chunk=100,
    line_overlap=10,
    max_char_limit=2500
)
```

### 3. Build & Query the FAISS Vector Store

```python
from indexer import build_repo_index, query_codebase

# Build and persist index
index = build_repo_index(chunks=chunks, save_dir="storage/faiss_index")

# Retrieve top-k relevant snippets
retrieved = query_codebase(
    query="How does JWT token verification work?",
    index=index,
    chunks=chunks,
    top_k=5
)
```

### 4. Grounded Q&A with Exact Line Citations (Gemini)

```python
from rag import answer_repo_query

# Automatically reads GEMINI_API_KEY from environment / .env
result = answer_repo_query(
    query="Explain the authentication workflow.",
    retrieved_chunks=retrieved,
    model="gemini-2.5-flash"
)

print("Answer:\n", result["answer"])
print("Citations:\n", result["citations"])
```

### 5. Trace Code Flow (Mermaid.js)

```python
from rag import trace_query_flow

# Generate Mermaid.js flowchart
mermaid_diagram = trace_query_flow(
    entry_query="Trace authentication and token validation flow",
    retrieved_chunks=retrieved,
    model="gemini-2.5-flash"
)

print(mermaid_diagram)
```

---

## ⚙️ Core Architecture

```text
                  +-----------------------+
                  |  Git Repo / ZIP File  |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |   RepositoryParser    |
                  | (AST / Line Splitter) |
                  +-----------+-----------+
                              |
                              v CodeSymbol objects
                  +-----------------------+
                  |      CodeChunker      |
                  |  (Contextual Headers) |
                  +-----------+-----------+
                              |
                              v Formatted chunks
                  +-----------------------+
                  |      VectorStore      |
                  | (SentenceTransformer) |
                  +-----------+-----------+
                              |
                              v Dense Embeddings (384-d)
                  +-----------------------+
                  |      FAISS Index      |
                  |  (Inner Product / IP) |
                  +-----------+-----------+
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
+-----------------------+           +-----------------------+
|  Dense Semantic Search|           |    Storage on Disk    |
| (query_codebase top_k)|           |  index.faiss + chunks |
+-----------+-----------+           +-----------------------+
            |
            v Top-K Context Chunks
+-----------------------------------------------+
|             Google Gemini Engine              |
|        (Server-Side GEMINI_API_KEY)           |
+-----------------------+-----------------------+
            |                                   |
            v                                   v
+-----------------------+           +-----------------------+
|   Citation RAG Engine |           |  Code Flow Visualizer |
|  Exact File:Line-Span |           |   Mermaid.js Flowchart|
+-----------+-----------+           +-----------+-----------+
            |                                   |
            +-----------------+-----------------+
                              |
                              v
                  +-----------------------+
                  | Streamlit Dashboard   |
                  | (Chat, Flow, AST Tree)|
                  +-----------------------+
```

---

## 🧪 Running Tests

### Part 1: Indexer Integration Tests
Tests AST symbol extraction, ZipSlip protection, semantic chunking, and FAISS indexing/retrieval:
```bash
python test_indexer.py
```

### Part 2: RAG & Flow Integration Tests
Tests prompt formatting, regex citation extraction, Mermaid flowchart generation, deterministic offline fallback, and unconfigured key error handling:
```bash
python test_rag.py
```

---

## 🛣️ Roadmap

- [x] AST-based Python code parser & multi-file scanner
- [x] Safe ZIP file extraction with ZipSlip defense
- [x] Contextual metadata header chunking with overlap handling
- [x] FAISS index generation, persistence, and dense retrieval
- [x] Gemini API integration (`google-genai`) for answer generation with code citations
- [x] Mermaid.js architectural code flow visualizer
- [x] Streamlit web UI for interactive repository upload, indexing, chat, and AST inspection
- [x] Server-side secure `GEMINI_API_KEY` handling with `.env` / `.env.example`

---

## 📄 License

This project is licensed under the MIT License.
