"""
GitRAG — Repository Intelligence Engine
Interactive Streamlit Application.

Provides:
1. Repository ingestion (Local folder or ZIP archive).
2. AST Symbol parsing, contextual chunking, and FAISS vector indexing.
3. Code Q&A Chat grounded with exact file and line-number citations via Gemini.
4. Mermaid.js Code Flow & Architecture Visualizer.
5. Repository Structure & Symbol Explorer.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# Support loading GEMINI_API_KEY from local .env during development
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from indexer.chunker import CodeChunker, chunk_symbols
from indexer.parser import (
    CodeSymbol,
    RepositoryParser,
    parse_repository,
    safe_extract_zip,
)
from indexer.vector_store import (
    DEFAULT_MODEL_NAME,
    DEFAULT_STORAGE_DIR,
    VectorStore,
    build_repo_index,
    load_repo_index,
    query_codebase,
)
from rag.citation_rag import (
    DEFAULT_GEMINI_MODEL,
    answer_repo_query,
    extract_citations,
    get_gemini_client,
)
from rag.code_flow import trace_query_flow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gitrag_app")

# Streamlit Page Configuration
st.set_page_config(
    page_title="GitRAG — Repository Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a polished, modern developer dashboard
st.markdown(
    """
    <style>
    /* Metric & Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.7));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .citation-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.85rem;
        font-family: monospace;
        margin: 2px 4px;
        font-weight: 600;
    }
    .symbol-pill {
        display: inline-block;
        background-color: #334155;
        color: #f1f5f9;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .score-badge {
        float: right;
        color: #10b981;
        font-weight: bold;
        font-size: 0.85rem;
    }
    /* Chat message spacing */
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "symbols" not in st.session_state:
    st.session_state.symbols = []
if "files_scanned" not in st.session_state:
    st.session_state.files_scanned = 0
if "repo_name" not in st.session_state:
    st.session_state.repo_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_flow_mermaid" not in st.session_state:
    st.session_state.latest_flow_mermaid = ""
if "latest_flow_query" not in st.session_state:
    st.session_state.latest_flow_query = ""


def render_mermaid(mermaid_code: str, height: int = 500) -> None:
    """Render a Mermaid.js diagram inside an isolated HTML iframe component."""
    escaped_code = mermaid_code.replace("\\", "\\\\").replace("`", "\\`")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 16px;
          background-color: transparent;
          color: #e2e8f0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 90vh;
        }}
        .mermaid-container {{
          width: 100%;
          display: flex;
          justify-content: center;
          overflow: auto;
        }}
        .mermaid {{
          max-width: 100%;
        }}
      </style>
      <script>
        document.addEventListener("DOMContentLoaded", function() {{
          mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            flowchart: {{
              useMaxWidth: true,
              htmlLabels: true,
              curve: 'basis'
            }}
          }});
        }});
      </script>
    </head>
    <body>
      <div class="mermaid-container">
        <pre class="mermaid">
{mermaid_code}
        </pre>
      </div>
    </body>
    </html>
    """
    components.html(html_content, height=height, scrolling=True)


# ==============================================================================
# SIDEBAR: CONFIGURATION & INGESTION
# ==============================================================================
with st.sidebar:
    st.title("⚡ GitRAG Engine")
    st.caption("Repository Intelligence & Citation RAG")

    st.markdown("---")
    st.subheader("⚙️ Model Configuration")

    model_option = st.selectbox(
        "Gemini Reasoning Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
        index=0,
        help="Select the Gemini model for synthesis and diagram generation.",
    )

    st.markdown("---")
    st.subheader("📦 Repository Source")

    source_tab1, source_tab2 = st.tabs(["Local Directory", "Upload ZIP"])

    source_type = "dir"
    repo_dir_path = ""
    uploaded_zip = None

    with source_tab1:
        repo_dir_path = st.text_input(
            "Local Folder Path",
            value=str(WORKSPACE_ROOT),
            help="Absolute or relative path to the codebase repository on disk.",
        )

    with source_tab2:
        uploaded_zip = st.file_uploader(
            "Upload Repository ZIP",
            type=["zip"],
            help="Upload a zipped repository archive.",
        )

    top_k_val = st.slider(
        "Retrieved Snippets (Top-K)",
        min_value=3,
        max_value=15,
        value=6,
        help="Number of most relevant code chunks to inject into the prompt context.",
    )

    # Index Repository Button
    index_btn = st.button("🚀 Index Repository", type="primary", use_container_width=True)

    if index_btn:
        with st.spinner("Analyzing repository, extracting AST symbols, and building FAISS index..."):
            try:
                symbols: List[CodeSymbol] = []
                repo_label = ""

                # Parse from ZIP or Directory
                if uploaded_zip is not None:
                    repo_label = uploaded_zip.name
                    zip_bytes = uploaded_zip.getvalue()
                    temp_dir = tempfile.mkdtemp(prefix="gitrag_zip_")
                    unpacked_path = safe_extract_zip(zip_bytes, temp_dir)
                    parser = RepositoryParser()
                    symbols = parser.scan_directory(unpacked_path)
                elif repo_dir_path.strip():
                    path_obj = Path(repo_dir_path.strip()).resolve()
                    if not path_obj.exists():
                        st.error(f"Directory does not exist: {path_obj}")
                    else:
                        repo_label = path_obj.name
                        parser = RepositoryParser()
                        symbols = parser.scan_directory(path_obj)
                else:
                    st.warning("Please specify a valid folder path or upload a ZIP file.")

                if symbols:
                    # Semantic chunking
                    chunker = CodeChunker(max_lines_per_chunk=100, line_overlap=10)
                    chunks = chunker.chunk_all(symbols)

                    # Build FAISS vector store
                    vector_store = VectorStore(model_name=DEFAULT_MODEL_NAME)
                    index = vector_store.build_index(chunks)

                    # Compute stats
                    scanned_files = len({s.file_path for s in symbols})

                    st.session_state.indexed = True
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.symbols = symbols
                    st.session_state.files_scanned = scanned_files
                    st.session_state.repo_name = repo_label

                    st.success(f"Successfully indexed **{repo_label}**!")
                else:
                    st.error("No supported code or documentation files were found to index.")

            except Exception as e:
                logger.exception("Indexing error: %s", e)
                st.error(f"Indexing failed: {e}")

    st.markdown("---")
    # Status & Metrics
    st.subheader("📊 Index Status")
    if st.session_state.indexed:
        st.success("🟢 Vector Store Active")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Files Parsed", st.session_state.files_scanned)
            st.metric("AST Symbols", len(st.session_state.symbols))
        with col_m2:
            st.metric("Indexed Chunks", len(st.session_state.chunks))
            st.metric("Vectors Total", st.session_state.index.ntotal if st.session_state.index else 0)
        st.caption(f"Embedding Model: `{DEFAULT_MODEL_NAME.split('/')[-1]}`")
    else:
        st.info("⚪ No repository indexed yet. Click 'Index Repository' above to start.")


# ==============================================================================
# MAIN PANEL: TABS
# ==============================================================================
tab_qa, tab_flow, tab_structure = st.tabs(
    ["💬 Code Q&A Chat", "🗺️ Code Flow Visualizer", "📂 Repository Structure"]
)

# ------------------------------------------------------------------------------
# TAB 1: CODE Q&A CHAT
# ------------------------------------------------------------------------------
with tab_qa:
    st.header("Codebase Question & Answer")
    st.caption("Ask questions about architecture, function logic, data flows, and dependencies with exact line citations.")

    if not st.session_state.indexed:
        st.warning("⚠️ Please index a repository in the sidebar first to begin querying.")
    else:
        # Display previous chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # If assistant message has citations, display interactive citation pills
                if msg.get("citations"):
                    st.markdown("**Exact Line Citations:**")
                    citation_html = " ".join(
                        f"<span class='citation-badge'>📄 {c['citation']}</span>"
                        for c in msg["citations"]
                    )
                    st.markdown(citation_html, unsafe_allow_html=True)

                # Expandable code snippets
                if msg.get("retrieved_snippets"):
                    with st.expander(f"🔍 Inspect {len(msg['retrieved_snippets'])} Retrieved Code Snippets"):
                        for i, snip in enumerate(msg["retrieved_snippets"], 1):
                            meta = snip.get("metadata", {})
                            fpath = meta.get("file_path", "unknown")
                            start_l = meta.get("start_line", 1)
                            end_l = meta.get("end_line", 1)
                            sym = meta.get("symbol_name", "")
                            stype = meta.get("type", "code")
                            score = snip.get("score", 0.0)

                            st.markdown(
                                f"**#{i} — `{fpath}`** (Lines {start_l}–{end_l}) | `{stype}: {sym}` "
                                f"<span class='score-badge'>Similarity: {score:.3f}</span>",
                                unsafe_allow_html=True,
                            )
                            # Code content
                            code_text = snip.get("text", "")
                            if "\n\n" in code_text:
                                code_text = code_text.split("\n\n", 1)[1]
                            lang = "python" if fpath.endswith(".py") else "text"
                            st.code(code_text, language=lang)
                            st.divider()

        # Chat user input
        user_query = st.chat_input("Ask a question about the repository (e.g., 'How does the parser handle AST symbols?')...")

        if user_query:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Retrieve relevant chunks from FAISS
            with st.spinner("Searching codebase and synthesizing answer with Gemini..."):
                retrieved = query_codebase(
                    query=user_query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    top_k=top_k_val,
                )

                # Generate Answer via Gemini (server-side environment key)
                rag_result = answer_repo_query(
                    query=user_query,
                    retrieved_chunks=retrieved,
                    model=model_option,
                )

                # Generate or update Code Flow diagram for Tab 2
                flow_mermaid = trace_query_flow(
                    entry_query=user_query,
                    retrieved_chunks=retrieved,
                    model=model_option,
                )
                st.session_state.latest_flow_mermaid = flow_mermaid
                st.session_state.latest_flow_query = user_query

                answer_text = rag_result.get("answer", "")
                citations = rag_result.get("citations", [])

                # Append assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations,
                    "retrieved_snippets": retrieved,
                }
                st.session_state.chat_history.append(assistant_msg)

                # Display response immediately
                with st.chat_message("assistant"):
                    st.markdown(answer_text)

                    if citations:
                        st.markdown("**Exact Line Citations:**")
                        citation_html = " ".join(
                            f"<span class='citation-badge'>📄 {c['citation']}</span>"
                            for c in citations
                        )
                        st.markdown(citation_html, unsafe_allow_html=True)

                    if retrieved:
                        with st.expander(f"🔍 Inspect {len(retrieved)} Retrieved Code Snippets"):
                            for i, snip in enumerate(retrieved, 1):
                                meta = snip.get("metadata", {})
                                fpath = meta.get("file_path", "unknown")
                                start_l = meta.get("start_line", 1)
                                end_l = meta.get("end_line", 1)
                                sym = meta.get("symbol_name", "")
                                stype = meta.get("type", "code")
                                score = snip.get("score", 0.0)

                                st.markdown(
                                    f"**#{i} — `{fpath}`** (Lines {start_l}–{end_l}) | `{stype}: {sym}` "
                                    f"<span class='score-badge'>Similarity: {score:.3f}</span>",
                                    unsafe_allow_html=True,
                                )
                                code_text = snip.get("text", "")
                                if "\n\n" in code_text:
                                    code_text = code_text.split("\n\n", 1)[1]
                                lang = "python" if fpath.endswith(".py") else "text"
                                st.code(code_text, language=lang)
                                st.divider()


# ------------------------------------------------------------------------------
# TAB 2: CODE FLOW VISUALIZER
# ------------------------------------------------------------------------------
with tab_flow:
    st.header("Code Execution & Architecture Flow")
    st.caption("Visual caller-callee sequences and architectural paths generated from retrieved code context.")

    if not st.session_state.indexed:
        st.info("💡 Index a repository first to trace code flows.")
    else:
        # User can trace a custom flow or use the latest query
        flow_col1, flow_col2 = st.columns([4, 1])
        with flow_col1:
            custom_flow_query = st.text_input(
                "Execution Flow / Architecture Entry Query:",
                value=st.session_state.latest_flow_query or "Trace the complete indexing and retrieval execution flow",
                help="Describe what execution path, call sequence, or subsystem flow you want to visualize.",
            )
        with flow_col2:
            st.write("")  # Alignment spacer
            st.write("")
            generate_flow_btn = st.button("🔄 Generate Flow", use_container_width=True)

        if generate_flow_btn and custom_flow_query.strip():
            with st.spinner("Retrieving components and tracing execution flow..."):
                retrieved_flow_chunks = query_codebase(
                    query=custom_flow_query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    top_k=top_k_val,
                )
                generated_mermaid = trace_query_flow(
                    entry_query=custom_flow_query,
                    retrieved_chunks=retrieved_flow_chunks,
                    model=model_option,
                )
                st.session_state.latest_flow_mermaid = generated_mermaid
                st.session_state.latest_flow_query = custom_flow_query

        # Render diagram if available
        if st.session_state.latest_flow_mermaid:
            st.subheader(f"Flow Diagram: {st.session_state.latest_flow_query}")
            render_mermaid(st.session_state.latest_flow_mermaid, height=520)

            with st.expander("📝 View Raw Mermaid.js Definition"):
                st.code(st.session_state.latest_flow_mermaid, language="mermaid")
        else:
            st.info("Enter a query and click **'Generate Flow'** or ask a question in the Chat tab to view the execution diagram.")


# ------------------------------------------------------------------------------
# TAB 3: REPOSITORY STRUCTURE
# ------------------------------------------------------------------------------
with tab_structure:
    st.header("Repository AST & Module Explorer")
    st.caption("Inspect parsed file hierarchy, classes, functions, methods, and line counts.")

    if not st.session_state.indexed or not st.session_state.symbols:
        st.info("Index a repository to explore its module structure.")
    else:
        # Group symbols by file path
        symbols_by_file: Dict[str, List[CodeSymbol]] = {}
        for sym in st.session_state.symbols:
            symbols_by_file.setdefault(sym.file_path, []).append(sym)

        filter_term = st.text_input("Filter files or symbols by name:", "")

        # Summary statistics
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Files", len(symbols_by_file))
        with col_s2:
            st.metric("Total Symbols", len(st.session_state.symbols))
        with col_s3:
            total_classes = sum(1 for s in st.session_state.symbols if s.symbol_type == "class")
            total_funcs = sum(1 for s in st.session_state.symbols if s.symbol_type in {"function", "method"})
            st.metric("Classes / Functions", f"{total_classes} / {total_funcs}")

        st.markdown("---")

        for fpath, file_symbols in sorted(symbols_by_file.items()):
            # Filter check
            if filter_term:
                fpath_matches = filter_term.lower() in fpath.lower()
                symbol_matches = any(filter_term.lower() in s.name.lower() for s in file_symbols)
                if not (fpath_matches or symbol_matches):
                    continue

            # File expander
            max_end = max((s.end_line for s in file_symbols), default=0)
            with st.expander(f"📄 **{fpath}** ({len(file_symbols)} symbols, ~{max_end} lines)"):
                for sym in file_symbols:
                    if filter_term and filter_term.lower() not in sym.name.lower() and filter_term.lower() not in fpath.lower():
                        continue

                    # Type badge color
                    color_map = {
                        "class": "#a855f7",
                        "function": "#3b82f6",
                        "method": "#06b6d4",
                        "module": "#64748b",
                        "doc": "#10b981",
                    }
                    badge_color = color_map.get(sym.symbol_type, "#6b7280")

                    st.markdown(
                        f"<span style='background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold;'>{sym.symbol_type.upper()}</span> "
                        f"**`{sym.name}`** &nbsp; <span style='color: #94a3b8; font-size: 0.85rem;'>(Lines {sym.start_line}–{sym.end_line})</span>",
                        unsafe_allow_html=True,
                    )

                    # Show imports if present
                    if sym.imports:
                        st.caption(f"Imports: {', '.join(sym.imports[:5])}{'...' if len(sym.imports) > 5 else ''}")

                    # Code preview toggle
                    with st.expander(f"View code: {sym.name}", expanded=False):
                        lang = "python" if fpath.endswith(".py") else "text"
                        st.code(sym.code, language=lang)
                    st.write("")
