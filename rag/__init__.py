"""
GitRAG RAG module.

Provides grounded repository question-answering with exact file and line-number citations,
and execution/architectural flow diagram tracing via Google Gemini.
"""

from .citation_rag import (
    DEFAULT_GEMINI_MODEL,
    answer_repo_query,
    extract_citations,
    format_context_for_prompt,
    get_gemini_client,
)
from .code_flow import (
    build_fallback_flowchart,
    sanitize_mermaid_code,
    trace_query_flow,
)

__all__ = [
    "answer_repo_query",
    "trace_query_flow",
    "extract_citations",
    "format_context_for_prompt",
    "get_gemini_client",
    "DEFAULT_GEMINI_MODEL",
    "build_fallback_flowchart",
    "sanitize_mermaid_code",
]
