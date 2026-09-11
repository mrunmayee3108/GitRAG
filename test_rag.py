"""
Comprehensive test suite for GitRAG Part 2 (RAG & Code Flow).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from rag.citation_rag import (
    extract_citations,
    format_context_for_prompt,
    answer_repo_query,
)
from rag.code_flow import (
    build_fallback_flowchart,
    sanitize_mermaid_code,
    trace_query_flow,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def test_context_formatter() -> None:
    logging.info("Testing format_context_for_prompt...")
    sample_chunks = [
        {
            "text": "File: backend/auth.py (Lines 10-30) | Type: function | Name: verify_token\n\ndef verify_token(token):\n    return True",
            "metadata": {
                "file_path": "backend/auth.py",
                "start_line": 10,
                "end_line": 30,
                "symbol_name": "verify_token",
                "type": "function",
                "imports": ["import jwt"],
            },
            "score": 0.8521,
        }
    ]

    formatted = format_context_for_prompt(sample_chunks)
    assert "File: backend/auth.py (Lines 10-30)" in formatted, "File header missing from context"
    assert "verify_token" in formatted, "Symbol missing from context"
    assert "0.8521" in formatted, "Score missing from context"
    logging.info("[PASSED] format_context_for_prompt verification.")


def test_citation_extraction() -> None:
    logging.info("Testing extract_citations...")
    sample_text = (
        "The authentication workflow begins in `login` [backend/auth.py:15-35] which validates credentials "
        "and issues a token via `create_token` [backend/tokens.py:40-65]. Single line citation [core.py:12]."
    )
    retrieved_chunks = [
        {
            "metadata": {
                "file_path": "backend/auth.py",
                "start_line": 15,
                "end_line": 35,
                "symbol_name": "login",
            }
        },
        {
            "metadata": {
                "file_path": "backend/tokens.py",
                "start_line": 40,
                "end_line": 65,
                "symbol_name": "create_token",
            }
        },
    ]

    citations = extract_citations(sample_text, retrieved_chunks)
    assert len(citations) == 3, f"Expected 3 citations, found {len(citations)}"

    first = citations[0]
    assert first["file_path"] == "backend/auth.py"
    assert first["start_line"] == 15
    assert first["end_line"] == 35
    assert first["symbol_name"] == "login"

    third = citations[2]
    assert third["file_path"] == "core.py"
    assert third["start_line"] == 12
    assert third["end_line"] == 12
    logging.info("[PASSED] extract_citations verification.")


def test_mermaid_sanitization_and_fallback() -> None:
    logging.info("Testing Mermaid sanitization and fallback generation...")
    raw_llm_mermaid = """```mermaid
graph TD
  A["User Query"] --> B["auth.py: verify()"]
  B --> C["db.py: find()"]
```"""
    cleaned = sanitize_mermaid_code(raw_llm_mermaid)
    assert not cleaned.startswith("```"), "Mermaid markdown fence not removed"
    assert cleaned.startswith("graph TD"), "Missing graph TD header"

    # Test fallback flowchart
    sample_chunks = [
        {
            "metadata": {
                "file_path": "indexer/parser.py",
                "start_line": 1,
                "end_line": 100,
                "symbol_name": "RepositoryParser",
                "type": "class",
            }
        },
        {
            "metadata": {
                "file_path": "indexer/chunker.py",
                "start_line": 19,
                "end_line": 140,
                "symbol_name": "CodeChunker",
                "type": "class",
            }
        },
    ]

    flow = trace_query_flow("How does indexing work?", sample_chunks)
    assert "flowchart TD" in flow or "graph TD" in flow, "Invalid diagram header"
    assert "indexer/parser.py" in flow, "File missing from flowchart"
    assert "CodeChunker" in flow, "Symbol missing from flowchart"
    logging.info("[PASSED] Mermaid flow verification.")


def test_answer_repo_query_without_api_key() -> None:
    logging.info("Testing answer_repo_query graceful error handling when API key is not configured...")
    import os
    sample_chunks = [
        {
            "text": "File: test.py (Lines 1-5)\nx = 1",
            "metadata": {"file_path": "test.py", "start_line": 1, "end_line": 5, "symbol_name": "test"},
        }
    ]
    # Temporarily unset GEMINI_API_KEY to test unconfigured server environment
    orig_key = os.environ.pop("GEMINI_API_KEY", None)
    try:
        result = answer_repo_query(
            query="What is x?",
            retrieved_chunks=sample_chunks,
        )
        assert "answer" in result, "Missing answer key in result dict"
        assert "citations" in result, "Missing citations key in result dict"
        assert isinstance(result["citations"], list), "citations must be a list"
        assert "GEMINI_API_KEY" in result["answer"] or "Configuration Error" in result["answer"], "Missing descriptive config error"
    finally:
        if orig_key is not None:
            os.environ["GEMINI_API_KEY"] = orig_key
    logging.info("[PASSED] answer_repo_query graceful handling verified.")


def main() -> None:
    logging.info("==================================================")
    logging.info("STARTING GITRAG PART 2 INTEGRATION TEST SUITE")
    logging.info("==================================================")
    test_context_formatter()
    test_citation_extraction()
    test_mermaid_sanitization_and_fallback()
    test_answer_repo_query_without_api_key()
    logging.info("==================================================")
    logging.info("ALL GITRAG PART 2 TESTS PASSED SUCCESSFULLY! ")
    logging.info("==================================================")


if __name__ == "__main__":
    main()
