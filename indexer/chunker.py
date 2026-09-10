"""
Semantic chunker module for GitRAG.

Transforms parsed CodeSymbol objects into structured text chunks with
file and symbol metadata headers designed to optimize embedding representations.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .parser import CodeSymbol

logger = logging.getLogger(__name__)


class CodeChunker:
    """Chunks CodeSymbol instances into structured chunks with contextual headers."""

    def __init__(
        self,
        max_lines_per_chunk: int = 100,
        line_overlap: int = 10,
        max_char_limit: int = 2500,
    ) -> None:
        """
        Initialize chunker configuration.

        Args:
            max_lines_per_chunk: Maximum lines for a single chunk before splitting.
            line_overlap: Overlap lines between adjacent chunks of a large symbol.
            max_char_limit: Maximum character limit for a chunk body.
        """
        self.max_lines_per_chunk = max_lines_per_chunk
        self.line_overlap = line_overlap
        self.max_char_limit = max_char_limit

    @staticmethod
    def format_header(
        file_path: str,
        start_line: int,
        end_line: int,
        symbol_type: str,
        symbol_name: str,
    ) -> str:
        """
        Construct the standardized metadata header required for embedding context.

        Format:
            File: {file_path} (Lines {start_line}-{end_line}) | Type: {symbol_type} | Name: {symbol_name}
        """
        return f"File: {file_path} (Lines {start_line}-{end_line}) | Type: {symbol_type} | Name: {symbol_name}"

    def _create_chunk_dict(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        symbol_type: str,
        symbol_name: str,
        code_content: str,
        imports: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Format chunk text and metadata dictionary."""
        header = self.format_header(file_path, start_line, end_line, symbol_type, symbol_name)
        text = f"{header}\n\n{code_content}".strip()

        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": symbol_name,
            "type": symbol_type,
            "imports": list(imports or []),
        }

        return {
            "text": text,
            "metadata": metadata,
        }

    def chunk_symbol(self, symbol: CodeSymbol) -> List[Dict[str, Any]]:
        """
        Process a single CodeSymbol into one or more chunks.

        If the symbol code is within limits, returns a single chunk.
        If oversized, breaks it into overlapping line windows while updating
        start_line and end_line in both header and metadata.
        """
        lines = symbol.code.splitlines()
        total_lines = len(lines)

        # If empty or small enough, return directly
        if total_lines <= self.max_lines_per_chunk and len(symbol.code) <= self.max_char_limit:
            return [
                self._create_chunk_dict(
                    file_path=symbol.file_path,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    symbol_type=symbol.symbol_type,
                    symbol_name=symbol.name,
                    code_content=symbol.code,
                    imports=symbol.imports,
                )
            ]

        chunks: List[Dict[str, Any]] = []
        step = max(1, self.max_lines_per_chunk - self.line_overlap)

        for idx in range(0, total_lines, step):
            slice_lines = lines[idx : idx + self.max_lines_per_chunk]
            if not slice_lines:
                continue

            chunk_start_line = symbol.start_line + idx
            chunk_end_line = min(symbol.start_line + idx + len(slice_lines) - 1, symbol.end_line)
            chunk_code = "\n".join(slice_lines)

            # Sub-chunk name identifier if split
            sub_name = f"{symbol.name} [part {len(chunks) + 1}]" if total_lines > self.max_lines_per_chunk else symbol.name

            chunks.append(
                self._create_chunk_dict(
                    file_path=symbol.file_path,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    symbol_type=symbol.symbol_type,
                    symbol_name=sub_name,
                    code_content=chunk_code,
                    imports=symbol.imports,
                )
            )

        return chunks

    def chunk_all(self, symbols: List[CodeSymbol]) -> List[Dict[str, Any]]:
        """Chunk a list of CodeSymbol objects."""
        all_chunks: List[Dict[str, Any]] = []
        for sym in symbols:
            try:
                chunks = self.chunk_symbol(sym)
                all_chunks.extend(chunks)
            except Exception as err:
                logger.error("Failed to chunk symbol %s: %s", getattr(sym, "name", "unknown"), err)
        return all_chunks


def chunk_symbols(
    symbols: List[CodeSymbol],
    max_lines_per_chunk: int = 100,
    line_overlap: int = 10,
    max_char_limit: int = 2500,
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk a list of CodeSymbol instances.

    Args:
        symbols: List of CodeSymbol objects.
        max_lines_per_chunk: Maximum number of lines per chunk.
        line_overlap: Number of overlapping lines between consecutive sub-chunks.
        max_char_limit: Character limit threshold for symbol splitting.

    Returns:
        List of chunk dictionaries with 'text' and 'metadata'.
    """
    chunker = CodeChunker(
        max_lines_per_chunk=max_lines_per_chunk,
        line_overlap=line_overlap,
        max_char_limit=max_char_limit,
    )
    return chunker.chunk_all(symbols)
