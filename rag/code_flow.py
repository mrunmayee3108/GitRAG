"""
Code Flow and Execution Path Tracer for GitRAG.

Traces architectural paths, caller-callee sequences, and component dependencies
across retrieved codebase snippets and returns a clean, renderable Mermaid.js diagram.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

from .citation_rag import DEFAULT_GEMINI_MODEL, format_context_for_prompt, get_gemini_client

logger = logging.getLogger(__name__)

MERMAID_SYSTEM_PROMPT = """You are a Principal Software Architect and Code Intelligence System.
Your task is to analyze retrieved code snippets and an entry query to produce a clean, valid, and visually organized Mermaid.js flowchart representing the execution flow or architectural component relationships.

STRICT MERMAID GENERATION RULES:
1. Valid Mermaid Syntax: Start the diagram with `graph TD` or `flowchart TD`.
2. Safe Labels: ALWAYS enclose node text in double quotes inside brackets:
   - Correct: `Node1["auth.py: verify_token()"] --> Node2["db.py: get_user()"]`
   - Incorrect: `Node1[auth.py: verify_token()]` (unquoted colons/parentheses cause parse errors)
3. Clarity: Show the direction of data or control flow from top-level entry point or caller to callees or downstream services.
4. Edge Labels: Where helpful, label edges with actions: `-->|validates|` or `-->|returns User|`.
5. Modularity: Group functions by file/module using `subgraph "module_name.py"` when multiple functions belong to the same module.
6. Purity: Output ONLY the Mermaid diagram code. Do NOT wrap in markdown fences (` ``` `), and do NOT add any markdown explanations before or after.
"""


def sanitize_mermaid_code(raw_text: str) -> str:
    """
    Sanitize and validate raw Mermaid text from LLM response.

    Strips code fences, removes extraneous text, and ensures valid entry directive.

    Args:
        raw_text: String output potentially containing markdown or backticks.

    Returns:
        Clean Mermaid diagram string.
    """
    text = raw_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (e.g. ```mermaid or ```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find the start of the mermaid diagram
    graph_keywords = ("graph TD", "graph LR", "graph TB", "flowchart TD", "flowchart LR", "flowchart TB", "sequenceDiagram")
    start_idx = -1
    for kw in graph_keywords:
        pos = text.find(kw)
        if pos != -1:
            if start_idx == -1 or pos < start_idx:
                start_idx = pos

    if start_idx != -1:
        text = text[start_idx:].strip()

    # If the text is empty or doesn't have graph keyword, default header
    if not any(text.startswith(kw) for kw in graph_keywords):
        text = f"graph TD\n{text}"

    return text


def build_fallback_flowchart(entry_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Deterministic fallback generator to construct a clean Mermaid flowchart
    from retrieved chunk metadata and symbols when Gemini is unavailable or offline.

    Args:
        entry_query: Developer question or flow prompt.
        retrieved_chunks: List of retrieved chunk dictionaries.

    Returns:
        Valid Mermaid diagram string.
    """
    if not retrieved_chunks:
        return (
            "graph TD\n"
            '  Query["Query: ' + entry_query.replace('"', "'")[:40] + '..."]\n'
            '  Notice["No code snippets available to trace flow"]\n'
            "  Query --> Notice\n"
        )

    lines: List[str] = ["flowchart TD"]
    lines.append(f'  Query(["Search Query: {entry_query[:35]}..."])')

    # Group chunks by file path
    file_to_symbols: Dict[str, List[Dict[str, Any]]] = {}
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk.get("metadata", {})
        fpath = meta.get("file_path", f"file_{i}.py")
        file_to_symbols.setdefault(fpath, []).append(meta)

    node_counter = 0
    prev_node_id: Optional[str] = None
    first_node_id: Optional[str] = None

    for f_idx, (fpath, metas) in enumerate(file_to_symbols.items()):
        subgraph_id = f"sub_{f_idx}"
        clean_fpath = fpath.replace("\\", "/").replace('"', "'")
        lines.append(f'  subgraph {subgraph_id} ["{clean_fpath}"]')

        for meta in metas:
            node_counter += 1
            node_id = f"N{node_counter}"
            s_name = meta.get("symbol_name", "snippet").replace('"', "'")
            s_type = meta.get("type", "symbol")
            s_lines = f"L{meta.get('start_line', 1)}-L{meta.get('end_line', 1)}"

            lines.append(f'    {node_id}["{s_name} ({s_type})<br/><i>{s_lines}</i>"]')

            if first_node_id is None:
                first_node_id = node_id

            if prev_node_id and prev_node_id != node_id:
                # Link adjacent nodes in flow
                lines.append(f"    {prev_node_id} -.->|calls/references| {node_id}")

            prev_node_id = node_id

        lines.append("  end")

    if first_node_id:
        lines.append(f"  Query -->|analyzes| {first_node_id}")

    return "\n".join(lines)


def trace_query_flow(
    entry_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    **kwargs: Any,
) -> str:
    """
    Trace caller-callee sequences or architectural paths across files,
    returning a clean Mermaid.js diagram string.

    The Gemini client is initialized server-side using the GEMINI_API_KEY environment variable.

    Args:
        entry_query: Entry point query or description of the execution flow to trace.
        retrieved_chunks: Retrieved code snippets with metadata from vector retrieval.
        client: Optional pre-configured genai.Client instance.
        model: Gemini model identifier.

    Returns:
        A valid Mermaid.js diagram string (e.g. `graph TD; ...`).
    """
    if not retrieved_chunks:
        return build_fallback_flowchart(entry_query, [])

    # Try resolving client server-side; if missing key, fallback immediately
    try:
        active_client = client or get_gemini_client()
    except Exception as err:
        logger.info("Using deterministic fallback flowchart (Gemini client not initialized: %s)", err)
        return build_fallback_flowchart(entry_query, retrieved_chunks)

    context_text = format_context_for_prompt(retrieved_chunks)

    prompt = (
        f"You are given the following codebase snippets retrieved for the flow query:\n\n"
        f"{context_text}\n\n"
        f"FLOW TRACE QUERY:\n{entry_query}\n\n"
        f"Generate a Mermaid.js diagram (`graph TD` or `flowchart TD`) showing the step-by-step "
        f"execution path, caller-callee sequence, and component interactions across the files. "
        f"Ensure every node label with special characters is enclosed in quotes: `Node[\"name\"]`. "
        f"Output ONLY the Mermaid code."
    )

    try:
        config = types.GenerateContentConfig(
            system_instruction=MERMAID_SYSTEM_PROMPT,
            temperature=0.1,
        )

        response = active_client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        raw_output = response.text or ""
        cleaned_diagram = sanitize_mermaid_code(raw_output)

        # Basic sanity check on length and syntax
        if "-->" in cleaned_diagram or "---" in cleaned_diagram:
            return cleaned_diagram

        # If model returned something invalid, fallback
        logger.warning("Model response did not contain valid Mermaid connections. Falling back.")
        return build_fallback_flowchart(entry_query, retrieved_chunks)

    except APIError as api_err:
        logger.error("Gemini API error during trace_query_flow: %s. Using fallback.", api_err)
        return build_fallback_flowchart(entry_query, retrieved_chunks)
    except Exception as err:
        logger.error("Unexpected error during trace_query_flow: %s. Using fallback.", err)
        return build_fallback_flowchart(entry_query, retrieved_chunks)
