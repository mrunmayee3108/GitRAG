# GitRAG 🔍

A high-performance codebase intelligence and Retrieval-Augmented Generation (RAG) engine designed to index, search, and understand software repositories with symbol-level precision and line-number accuracy.

---

## 🌟 Overview

**GitRAG** transforms source code repositories and documentation into an AI-ready semantic vector database. Unlike generic text chunkers that blindly slice code and lose structure, GitRAG uses **AST (Abstract Syntax Tree)** parsing to extract meaningful semantic symbols (classes, functions, methods, docstrings, and imports) while maintaining exact line-level references.

Retrieved code chunks are enriched with contextual metadata headers, embedded using dense vector representations, and queried in milliseconds via **FAISS**.

---

## ✨ Features

- 🌳 **AST-Powered Symbol Extraction**: Parses Python code into standalone functions, classes, and methods using Python's native `ast` module.
- 📑 **Multi-Language & Docs Support**: Supports Python (`.py`), JavaScript/TypeScript (`.js`, `.ts`, `.jsx`, `.tsx`), and Markdown (`.md`).
- 🛡️ **Zip-Slip Safe Archive Extraction**: Safely inspects and extracts uploaded `.zip` repositories while preventing path traversal vulnerabilities (`..` escape attacks).
- 🧩 **Context-Aware Semantic Chunking**: Enriches each chunk with a standardized header (`File: ... (Lines start-end) | Type: ... | Name: ...`) and handles oversized symbols with intelligent sliding-window overlap.
- ⚡ **FAISS Vector Search**: Builds L2-normalized Inner Product indices (equivalent to Cosine Similarity) for fast, dense semantic retrieval.
- 💾 **Index Persistence**: Easily serialize (`index.faiss` and `chunks.json`) and reload indices from disk.
- 🧪 **Self-Contained Test Suite**: Includes comprehensive integration tests (`test_indexer.py`) that validate parsing, chunking, indexing, and retrieval in isolated environments.

---

## 📁 Repository Structure

```text
GitRAG/
├── indexer/
│   ├── __init__.py          # Public package exports
│   ├── parser.py            # AST parsing, file scanning, and safe ZIP extraction
│   ├── chunker.py           # Contextual chunking with metadata headers
│   └── vector_store.py      # FAISS vector store & SentenceTransformer retrieval
├── requirements.txt         # Project dependencies
├── test_indexer.py          # End-to-end integration test suite
└── README.md                # Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ recommended
- `pip` or virtual environment manager (`venv` / `conda`)

### 2. Installation

Clone this repository and create a virtual environment:

```bash
# Clone the repository
git clone https://github.com/mrunmayee3108/GitRAG.git
cd GitRAG

# Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Dependencies

- **`sentence-transformers`**: Generates dense code and query embeddings (`sentence-transformers/all-MiniLM-L6-v2` by default).
- **`faiss-cpu`**: High-performance vector similarity search.
- **`numpy`**: Array manipulation for vector normalizations.
- **`google-genai`**: For Gemini LLM query synthesis and RAG answering.
- **`streamlit`**: For the interactive web dashboard.

---

## 💡 Quick Start & Usage

### 1. Parse a Repository or ZIP Archive

You can parse any local directory or ZIP archive directly into structured `CodeSymbol` objects:

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

Chunk symbols into embedding-ready text segments:

```python
from indexer import chunk_symbols

# Generates chunks with standardized metadata headers and sliding-window overlap
chunks = chunk_symbols(
    symbols,
    max_lines_per_chunk=100,
    line_overlap=10,
    max_char_limit=2500
)

# Example chunk structure:
# chunks[0]["text"]     -> "File: auth.py (Lines 10-25) | Type: function | Name: verify_token\n\ndef verify_token(..."
# chunks[0]["metadata"] -> {"file_path": "auth.py", "start_line": 10, "end_line": 25, "symbol_name": "verify_token", ...}
```

### 3. Build and Persist the FAISS Index

Encode chunks into dense vectors and persist them to disk:

```python
from indexer import build_repo_index

storage_directory = "storage/faiss_index"

# Builds FAISS index and writes index.faiss and chunks.json
index = build_repo_index(
    chunks=chunks,
    save_dir=storage_directory,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 4. Load Index and Query Codebase

Load an existing index from disk and run natural language or code searches:

```python
from indexer import load_repo_index, query_codebase

# Load persisted index & chunk metadata
index, chunks = load_repo_index(save_dir="storage/faiss_index")

# Perform semantic search
query = "How is user token verification handled?"
results = query_codebase(query=query, index=index, chunks=chunks, top_k=3)

for rank, match in enumerate(results, start=1):
    meta = match["metadata"]
    print(f"\n[Match #{rank}] Score: {match['score']:.4f}")
    print(f"Symbol: {meta['symbol_name']} ({meta['type']})")
    print(f"Location: {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']})")
    print("-" * 50)
    print(match["text"][:300] + "...")
```

---

## ⚙️ Core Architecture

```
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
             +---------------+---------------+
             |                               |
             v                               v
    [index.faiss]                    [chunks.json]
```

### Modules Breakdown

1. **`indexer/parser.py`**:
   - `RepositoryParser`: Scans directories while ignoring `.git`, `node_modules`, `__pycache__`, virtual environments, etc.
   - `PythonASTVisitor`: Walks Python AST trees to extract functions, classes, methods, imports, and docstrings with 1-indexed line boundaries.
   - `safe_extract_zip`: Guards against Zip-Slip path traversal vulnerabilities.
2. **`indexer/chunker.py`**:
   - `CodeChunker`: Prepends contextual headers to snippets so embeddings retain file path and symbol identity even when split across multiple windows.
3. **`indexer/vector_store.py`**:
   - `VectorStore`: Encapsulates embedding creation with `SentenceTransformer`, index generation via `faiss.IndexFlatIP`, disk serialization, and top-$k$ dense similarity retrieval.

---

## 🧪 Running Tests

GitRAG includes a self-contained test suite that automatically sets up a mock multi-file codebase, tests ZIP extraction, executes AST parsing, performs semantic chunking, and runs FAISS similarity queries:

```bash
python test_indexer.py
```

Expected output:
```text
[INFO] STARTING GITRAG CORE ENGINE INTEGRATION TEST SUITE
[INFO] --> Testing RepositoryParser on directory...
[INFO] [PASSED] RepositoryParser and AST extraction verification.
[INFO] --> Testing ZIP archive extraction and parsing...
[INFO] [PASSED] ZIP archive safe extraction verification.
[INFO] --> Testing CodeChunker...
[INFO] [PASSED] CodeChunker metadata and header formatting verification.
[INFO] --> Testing VectorStore index construction and persistence...
[INFO] --> Testing loading index from disk...
[INFO] --> Testing semantic search queries...
[INFO] [PASSED] FAISS index persistence and semantic query verification.
[INFO] ALL GITRAG INTEGRATION TESTS PASSED SUCCESSFULLY! 
```


## 🛣️ Roadmap

- [x] AST-based Python code parser & multi-file scanner
- [x] Safe ZIP file extraction with ZipSlip defense
- [x] Contextual metadata header chunking with overlap handling
- [x] FAISS index generation, persistence, and dense retrieval
- [ ] Gemini API integration (`google-genai`) for answer generation with code citations
- [ ] Streamlit web UI for interactive repository upload, indexing, and chat

---

## 📄 License

This project is licensed under the MIT License.
