"""
Citation RAG Engine for GitRAG.

Integrates FAISS retrieval with Google Gemini via the official google-genai SDK,
enforcing strict grounding to retrieved code context and producing exact file and
line-number citations.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from google import genai
from google.genai import types
from google.genai.errors import APIError

# Support GEMINI_API_KEY from local .env file during development
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Default Gemini model used for repository reasoning
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are GitRAG, a Principal Code Intelligence and Repository QA engine.
Your task is to answer technical and architectural questions about a codebase strictly and exclusively using the provided code snippets.

STRICT GROUNDING RULES:
1. Grounding: Answer ONLY using the facts, signatures, implementations, and comments directly present in the provided snippets. If the snippets do not contain enough information to answer completely, explicitly state what is missing or unknown. NEVER fabricate or extrapolate missing files, methods, or logic.
2. Citations: Every single technical claim, function reference, or implementation detail MUST be cited inline using the exact bracket notation: `[file_path:start_line-end_line]` or `[file_path:line_number]`.
   - Example: "The JWT token is verified in `verify_bearer_token` [auth/tokens.py:45-62] using HMAC SHA-256."
   - Example: "The class `VectorStore` initializes FAISS [indexer/vector_store.py:23-50]."
   - Never omit line numbers or file paths when citing code.
3. Accuracy: Reference symbol names, parameters, error types, and classes exactly as written in the snippets.
4. Response Format:
   - Provide a clear, developer-focused explanation in GitHub-flavored Markdown.
   - Use bold headers, concise bullet points, and code blocks for syntax.
   - At the end of your response, include a concise "### Referenced Citations" summary list with each file and line span cited.
"""


def get_gemini_client() -> genai.Client:
    """
    Initialize and return the official Google GenAI Client server-side.

    Reads GEMINI_API_KEY strictly from the environment variable.

    Returns:
        genai.Client instance.

    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not configured.
    """
    resolved_key = os.getenv("GEMINI_API_KEY")
    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please set GEMINI_API_KEY in your server environment or in a local .env file."
        )
    return genai.Client(api_key=resolved_key.strip())


def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved code chunks into structured context blocks for the LLM prompt.

    Args:
        retrieved_chunks: List of chunk dictionaries containing 'text', 'metadata',
                          and optional 'score'.

    Returns:
        Formatted context string with line numbers and file boundaries.
    """
    if not retrieved_chunks:
        return "No relevant code snippets were retrieved from the repository."

    context_parts: List[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "unknown_file")
        start_line = meta.get("start_line", 1)
        end_line = meta.get("end_line", 1)
        symbol_name = meta.get("symbol_name", "snippet")
        symbol_type = meta.get("type", "code")
        imports = meta.get("imports", [])
        score = chunk.get("score")

        # Determine code body
        code_body = chunk.get("text", "")
        # If the chunk text has the standard header, separate it cleanly
        header_prefix = f"File: {file_path}"
        if code_body.startswith(header_prefix) and "\n\n" in code_body:
            code_body = code_body.split("\n\n", 1)[1]

        score_info = f" | Similarity Score: {score:.4f}" if score is not None else ""
        import_info = f"\nImports: {', '.join(imports)}" if imports else ""

        snippet_block = (
            f"--- Snippet #{idx} ---\n"
            f"File: {file_path} (Lines {start_line}-{end_line})\n"
            f"Type: {symbol_type} | Symbol: {symbol_name}{score_info}{import_info}\n"
            f"```\n"
            f"{code_body}\n"
            f"```\n"
        )
        context_parts.append(snippet_block)

    return "\n".join(context_parts)


def extract_citations(text: str, retrieved_chunks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Parse inline citations from text and cross-reference with retrieved chunks.

    Matches patterns like [file.py:10-25] or [dir/file.py:42].

    Args:
        text: Response markdown text containing citations.
        retrieved_chunks: Optional list of retrieved chunks for metadata enrichment.

    Returns:
        List of unique citation dictionaries:
        [{
            "citation": "[file.py:10-25]",
            "file_path": "file.py",
            "start_line": 10,
            "end_line": 25,
            "symbol_name": "...",
        }]
    """
    citation_pattern = re.compile(
        r"\[([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+):(\d+)(?:-(\d+))?\]"
    )

    seen: Set[str] = set()
    citations: List[Dict[str, Any]] = []

    # Map file_path -> list of chunk metadatas for fast symbol lookup
    chunk_map: Dict[str, List[Dict[str, Any]]] = {}
    if retrieved_chunks:
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            fpath = meta.get("file_path")
            if fpath:
                normalized_fpath = fpath.replace("\\", "/")
                chunk_map.setdefault(normalized_fpath, []).append(meta)

    for match in citation_pattern.finditer(text):
        full_match = match.group(0)
        if full_match in seen:
            continue
        seen.add(full_match)

        raw_file = match.group(1).replace("\\", "/")
        start_line = int(match.group(2))
        end_line = int(match.group(3)) if match.group(3) else start_line

        # Look for matching symbol in retrieved metadata
        matching_symbol = ""
        for known_path, metas in chunk_map.items():
            if known_path.endswith(raw_file) or raw_file.endswith(known_path):
                for meta in metas:
                    s_start = meta.get("start_line", 0)
                    s_end = meta.get("end_line", 0)
                    if not (end_line < s_start or start_line > s_end):
                        matching_symbol = meta.get("symbol_name", "")
                        break
            if matching_symbol:
                break

        citations.append(
            {
                "citation": full_match,
                "file_path": raw_file,
                "start_line": start_line,
                "end_line": end_line,
                "symbol_name": matching_symbol,
            }
        )

    return citations


def answer_repo_query(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate a grounded repository answer with exact citations using Gemini.

    The Gemini client is initialized server-side from the GEMINI_API_KEY environment variable.

    Args:
        query: Developer's question regarding the codebase.
        retrieved_chunks: List of retrieved chunk dictionaries from VectorStore.
        client: Optional pre-configured genai.Client instance.
        model: Gemini model identifier (e.g. 'gemini-2.5-flash').
        temperature: Generation temperature (low for strict grounding).

    Returns:
        Dictionary containing:
        - 'answer': Markdown-formatted answer with inline citations.
        - 'citations': List of unique referenced files with line ranges.
        - 'retrieved_count': Number of context chunks provided.
        - 'model': The model name used for generation.
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant code snippets were found in the indexed repository for this query.",
            "citations": [],
            "retrieved_count": 0,
            "model": model,
        }

    # Resolve or create client server-side from environment
    try:
        active_client = client or get_gemini_client()
    except Exception as err:
        logger.error("Configuration error initializing Gemini client: %s", err)
        return {
            "answer": (
                f"**Configuration Error**: {err}\n\n"
                "Please configure `GEMINI_API_KEY` in your server environment or in a local `.env` file."
            ),
            "citations": [],
            "retrieved_count": len(retrieved_chunks),
            "model": model,
        }

    # Construct context and user prompt
    context_text = format_context_for_prompt(retrieved_chunks)

    prompt = (
        f"You are given the following retrieved code snippets from the codebase:\n\n"
        f"{context_text}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"Provide an accurate, grounded answer strictly based on the snippets above. "
        f"Cite every file and line range inline as `[file_path:start-end]`."
    )

    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        )

        response = active_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        answer_text = response.text or "No response generated by Gemini."
        citations = extract_citations(answer_text, retrieved_chunks=retrieved_chunks)

        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(retrieved_chunks),
            "model": model,
        }

    except APIError as api_err:
        logger.error("Gemini API Error during answer_repo_query: %s", api_err)
        return {
            "answer": (
                f"**Gemini API Error**: {api_err.message or api_err}\n\n"
                "Please check your `GEMINI_API_KEY` configuration and quota."
            ),
            "citations": [],
            "retrieved_count": len(retrieved_chunks),
            "model": model,
        }
    except Exception as err:
        logger.error("Unexpected error in answer_repo_query: %s", err)
        return {
            "answer": f"**An error occurred while communicating with Gemini**: {err}",
            "citations": [],
            "retrieved_count": len(retrieved_chunks),
            "model": model,
        }
