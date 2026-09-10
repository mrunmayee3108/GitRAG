"""
Self-contained integration test suite for GitRAG Indexer Engine.

Tests the complete pipeline:
1. Dynamic creation of a sample multi-file codebase with Python and Markdown files.
2. Compression into a ZIP archive to test archive extraction.
3. AST symbol extraction (classes, functions, methods, imports, docstrings, line numbers).
4. Semantic chunking with standardized contextual headers.
5. FAISS vector store indexing, persistence, and loading.
6. Dense semantic search verifying metadata and line number retention in top-k results.
7. Cleanup of temporary directories and artifacts.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Add workspace to sys.path so indexer is importable
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
    VectorStore,
    build_repo_index,
    load_repo_index,
    query_codebase,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("GitRAGTest")


def create_dummy_codebase(base_dir: Path) -> Path:
    """
    Dynamically construct a realistic mini codebase for testing.

    Includes:
    - calculator.py: Class with methods, imports, docstrings, type annotations.
    - auth.py: Functions, docstrings, error handling.
    - README.md: Documentation for repository overview.
    - utils/string_helper.py: Subdirectory module.
    - .git / __pycache__: Ignored directories that should be skipped.
    """
    project_dir = base_dir / "sample_repo"
    project_dir.mkdir(parents=True, exist_ok=True)

    # 1. calculator.py
    calc_code = '''"""
Scientific and basic arithmetic operations module.
"""

import math
from typing import List, Union

class Calculator:
    """A high-precision scientific calculator."""

    def __init__(self, precision: int = 4) -> None:
        """Initialize calculator with rounding precision."""
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers and return rounded result."""
        return round(a + b, self.precision)

    def calculate_factorial(self, n: int) -> int:
        """Compute the factorial of a non-negative integer using math.factorial."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative integers.")
        return math.factorial(n)

def compute_fibonacci(count: int) -> List[int]:
    """Generate a sequence of Fibonacci numbers up to count."""
    if count <= 0:
        return []
    sequence = [0, 1]
    while len(sequence) < count:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:count]
'''
    (project_dir / "calculator.py").write_text(calc_code, encoding="utf-8")

    # 2. auth.py
    auth_code = '''"""
Authentication and JWT token verification service.
"""

import os
import secrets
from typing import Optional

def generate_session_token(length: int = 32) -> str:
    """Generate a secure cryptographically random hex token."""
    return secrets.token_hex(length)

def verify_bearer_token(auth_header: Optional[str]) -> bool:
    """Validate Bearer authentication header against environment secret."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ", 1)[1]
    expected_token = os.environ.get("API_SECRET_KEY", "default_secret")
    return secrets.compare_digest(token, expected_token)
'''
    (project_dir / "auth.py").write_text(auth_code, encoding="utf-8")

    # 3. utils/string_helper.py
    utils_dir = project_dir / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)
    helper_code = '''"""
String formatting and text manipulation helpers.
"""

def slugify(text: str) -> str:
    """Convert arbitrary text into a URL-safe lowercase slug."""
    return "-".join(text.lower().strip().split())
'''
    (utils_dir / "string_helper.py").write_text(helper_code, encoding="utf-8")

    # 4. README.md
    readme_content = """# Sample Codebase

GitRAG test repository demonstrating multi-module parsing and indexing.

## Features
- **Calculator**: High-precision math operations and factorial calculation.
- **Authentication**: Secure token generation and header verification.
- **Utilities**: String formatting and slugification tools.
"""
    (project_dir / "README.md").write_text(readme_content, encoding="utf-8")

    # 5. Ignored folders to verify filter logic
    (project_dir / ".git").mkdir(exist_ok=True)
    (project_dir / ".git" / "config").write_text("git config dummy", encoding="utf-8")
    (project_dir / "__pycache__").mkdir(exist_ok=True)
    (project_dir / "__pycache__" / "calc.pyc").write_text("binary", encoding="utf-8")

    return project_dir


def create_dummy_zip(source_dir: Path, zip_dest: Path) -> Path:
    """Compress source directory into a ZIP archive."""
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zf.write(file_path, arcname)
    return zip_dest


def test_parser_and_ast(project_dir: Path) -> list[CodeSymbol]:
    """Test AST symbol extraction and directory scanning."""
    logger.info("--> Testing RepositoryParser on directory...")
    parser = RepositoryParser()
    symbols = parser.scan_directory(project_dir)

    assert len(symbols) > 0, "No symbols were extracted from the codebase."

    symbol_names = {s.name for s in symbols}
    symbol_types = {s.symbol_type for s in symbols}

    logger.info("Extracted %d symbols: %s", len(symbols), sorted(symbol_names))
    logger.info("Extracted symbol types: %s", sorted(symbol_types))

    # Validate essential symbols
    assert "Calculator" in symbol_names, "Class 'Calculator' was not extracted."
    assert "Calculator.add" in symbol_names, "Method 'Calculator.add' was not extracted."
    assert "Calculator.calculate_factorial" in symbol_names, "Method 'Calculator.calculate_factorial' was not extracted."
    assert "compute_fibonacci" in symbol_names, "Function 'compute_fibonacci' was not extracted."
    assert "generate_session_token" in symbol_names, "Function 'generate_session_token' was not extracted."
    assert "verify_bearer_token" in symbol_names, "Function 'verify_bearer_token' was not extracted."
    assert "slugify" in symbol_names, "Function 'slugify' in subdirectory was not extracted."
    assert "README.md" in symbol_names, "Document 'README.md' was not extracted."

    # Validate line numbers and imports
    factorial_sym = next(s for s in symbols if s.name == "Calculator.calculate_factorial")
    assert factorial_sym.start_line > 0, "Invalid start_line in Calculator.calculate_factorial"
    assert factorial_sym.end_line >= factorial_sym.start_line, "end_line must be >= start_line"
    assert "import math" in factorial_sym.imports, "Imports were not retained in symbol"
    assert "math.factorial" in factorial_sym.code, "Code snippet was not captured properly"
    assert factorial_sym.symbol_type == "method", "Expected symbol_type 'method'"

    # Verify that ignored files were not scanned
    for s in symbols:
        assert ".git" not in s.file_path, f"Ignored directory scanned: {s.file_path}"
        assert "__pycache__" not in s.file_path, f"Ignored directory scanned: {s.file_path}"

    logger.info("[PASSED] RepositoryParser and AST extraction verification.")
    return symbols


def test_zip_unpacking(zip_path: Path, temp_dir: Path) -> None:
    """Test safe extraction and parsing of a ZIP archive."""
    logger.info("--> Testing ZIP archive extraction and parsing...")
    extract_target = temp_dir / "unpacked_zip"
    symbols = parse_repository(zip_path, extract_target_dir=extract_target)

    assert len(symbols) > 0, "Failed to extract symbols from ZIP archive."
    symbol_names = {s.name for s in symbols}
    assert "Calculator" in symbol_names, "Missing Calculator class in ZIP extracted symbols."
    assert "verify_bearer_token" in symbol_names, "Missing verify_bearer_token in ZIP symbols."

    logger.info("[PASSED] ZIP archive safe extraction verification.")


def test_chunker(symbols: list[CodeSymbol]) -> list[dict]:
    """Test chunk generation and contextual header formatting."""
    logger.info("--> Testing CodeChunker...")
    chunks = chunk_symbols(symbols)

    assert len(chunks) >= len(symbols), "Chunk count should be >= symbol count."

    for chunk in chunks:
        assert "text" in chunk, "Chunk dictionary missing 'text' field."
        assert "metadata" in chunk, "Chunk dictionary missing 'metadata' field."

        meta = chunk["metadata"]
        assert "file_path" in meta, "Metadata missing 'file_path'."
        assert "start_line" in meta, "Metadata missing 'start_line'."
        assert "end_line" in meta, "Metadata missing 'end_line'."
        assert "symbol_name" in meta, "Metadata missing 'symbol_name'."
        assert "type" in meta, "Metadata missing 'type'."

        assert meta["start_line"] > 0, f"Invalid start_line: {meta['start_line']}"
        assert meta["end_line"] >= meta["start_line"], f"Invalid range: {meta['start_line']}-{meta['end_line']}"

        # Verify contextual header format:
        # File: {file_path} (Lines {start_line}-{end_line}) | Type: {symbol_type} | Name: {symbol_name}
        expected_header = (
            f"File: {meta['file_path']} (Lines {meta['start_line']}-{meta['end_line']}) "
            f"| Type: {meta['type']} | Name: {meta['symbol_name']}"
        )
        assert chunk["text"].startswith(expected_header), (
            f"Chunk text does not start with expected header!\nExpected: {expected_header}\nGot: {chunk['text'][:100]}"
        )

    logger.info("[PASSED] CodeChunker metadata and header formatting verification.")
    return chunks


def test_vector_store(chunks: list[dict], storage_dir: Path) -> None:
    """Test FAISS index building, disk serialization, and semantic retrieval."""
    logger.info("--> Testing VectorStore index construction and persistence...")
    index = build_repo_index(chunks, save_dir=str(storage_dir))

    assert index is not None, "Failed to create FAISS index."
    assert index.ntotal == len(chunks), f"FAISS vector count ({index.ntotal}) does not match chunk count ({len(chunks)})."

    # Verify files created on disk
    assert (storage_dir / "index.faiss").exists(), "index.faiss was not persisted."
    assert (storage_dir / "chunks.json").exists(), "chunks.json was not persisted."

    logger.info("--> Testing loading index from disk...")
    loaded_index, loaded_chunks = load_repo_index(save_dir=str(storage_dir))
    assert loaded_index.ntotal == index.ntotal, "Loaded index vector count mismatch."
    assert len(loaded_chunks) == len(chunks), "Loaded chunks count mismatch."

    logger.info("--> Testing semantic search queries...")

    # Query 1: Calculator factorial
    q1 = "calculate factorial of a number"
    results1 = query_codebase(q1, loaded_index, loaded_chunks, top_k=3)
    assert len(results1) > 0, "No results returned for query 1"
    logger.info("Query '%s' Top match: %s (score: %.4f)", q1, results1[0]["metadata"]["symbol_name"], results1[0]["score"])
    top_sym1 = results1[0]["metadata"]["symbol_name"]
    assert "factorial" in top_sym1.lower() or "calculator" in top_sym1.lower(), (
        f"Expected factorial or calculator in top result, got: {top_sym1}"
    )
    assert results1[0]["metadata"]["start_line"] > 0
    assert results1[0]["metadata"]["end_line"] >= results1[0]["metadata"]["start_line"]
    assert "calculator.py" in results1[0]["metadata"]["file_path"]

    # Query 2: Authentication and tokens
    q2 = "authenticate user and verify bearer token"
    results2 = query_codebase(q2, loaded_index, loaded_chunks, top_k=3)
    assert len(results2) > 0, "No results returned for query 2"
    logger.info("Query '%s' Top match: %s (score: %.4f)", q2, results2[0]["metadata"]["symbol_name"], results2[0]["score"])
    top_sym2 = results2[0]["metadata"]["symbol_name"]
    assert "token" in top_sym2.lower() or "auth" in results2[0]["metadata"]["file_path"].lower(), (
        f"Expected token or auth in top result, got: {top_sym2}"
    )
    assert results2[0]["metadata"]["start_line"] > 0
    assert "auth.py" in results2[0]["metadata"]["file_path"]

    # Query 3: Documentation overview
    q3 = "what is the overview and features of this repository?"
    results3 = query_codebase(q3, loaded_index, loaded_chunks, top_k=3)
    assert len(results3) > 0, "No results returned for query 3"
    logger.info("Query '%s' Top match: %s (score: %.4f)", q3, results3[0]["metadata"]["symbol_name"], results3[0]["score"])
    matched_files = [r["metadata"]["file_path"] for r in results3]
    assert any("README.md" in fp for fp in matched_files), "Expected README.md among top matches for repo overview query."

    logger.info("[PASSED] FAISS index persistence and semantic query verification.")


def main() -> None:
    """Run all end-to-end tests in an isolated temporary environment."""
    logger.info("==================================================")
    logger.info("STARTING GITRAG CORE ENGINE INTEGRATION TEST SUITE")
    logger.info("==================================================")

    temp_dir = Path(tempfile.mkdtemp(prefix="gitrag_test_"))
    try:
        # Step 1: Create dummy project
        project_dir = create_dummy_codebase(temp_dir)
        zip_path = temp_dir / "sample_repo.zip"
        create_dummy_zip(project_dir, zip_path)

        # Step 2: Test parser on directory
        symbols = test_parser_and_ast(project_dir)

        # Step 3: Test safe ZIP extraction
        test_zip_unpacking(zip_path, temp_dir)

        # Step 4: Test semantic chunker
        chunks = test_chunker(symbols)

        # Step 5: Test FAISS vector store
        storage_dir = temp_dir / "storage" / "faiss_index"
        test_vector_store(chunks, storage_dir)

        logger.info("==================================================")
        logger.info("ALL GITRAG INTEGRATION TESTS PASSED SUCCESSFULLY! ")
        logger.info("==================================================")

    finally:
        # Cleanup temporary test directory
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary test environment: %s", temp_dir)
        except Exception as cleanup_err:
            logger.warning("Failed to remove temp dir %s: %s", temp_dir, cleanup_err)


if __name__ == "__main__":
    main()
