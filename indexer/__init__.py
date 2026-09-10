"""
GitRAG Indexer package.

Provides AST parsing, semantic chunking, and vector search capabilities
for codebase intelligence.
"""

from .chunker import CodeChunker, chunk_symbols
from .parser import (
    IGNORED_DIRECTORIES,
    SUPPORTED_EXTENSIONS,
    CodeSymbol,
    PythonASTVisitor,
    RepositoryParser,
    parse_repository,
    safe_extract_zip,
)
from .vector_store import (
    DEFAULT_MODEL_NAME,
    DEFAULT_STORAGE_DIR,
    VectorStore,
    build_repo_index,
    load_repo_index,
    query_codebase,
)

__all__ = [
    "CodeSymbol",
    "PythonASTVisitor",
    "RepositoryParser",
    "safe_extract_zip",
    "parse_repository",
    "SUPPORTED_EXTENSIONS",
    "IGNORED_DIRECTORIES",
    "CodeChunker",
    "chunk_symbols",
    "VectorStore",
    "build_repo_index",
    "load_repo_index",
    "query_codebase",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_STORAGE_DIR",
]
