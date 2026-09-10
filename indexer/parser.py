"""
Codebase parser module for GitRAG.

Provides AST-based symbol extraction for Python and line-bounded content
extraction for documentation and web languages (.js, .ts, .jsx, .tsx, .md),
along with safe archive extraction and directory scanning.
"""

from __future__ import annotations

import ast
import io
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set, Union

logger = logging.getLogger(__name__)

# Default file extensions recognized by GitRAG
SUPPORTED_EXTENSIONS: Set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".md",
}

# Directories excluded from scanning
IGNORED_DIRECTORIES: Set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    ".tox",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "site-packages",
}


@dataclass
class CodeSymbol:
    """Represents an extracted code symbol (function, class, method, module, or document)."""

    name: str
    symbol_type: str  # 'function', 'class', 'method', 'module', 'doc'
    file_path: str
    start_line: int
    end_line: int
    code: str
    imports: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert symbol to dictionary representation."""
        return {
            "name": self.name,
            "symbol_type": self.symbol_type,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "imports": list(self.imports),
        }


class PythonASTVisitor(ast.NodeVisitor):
    """AST visitor to extract functions, classes, methods, imports, and docstrings."""

    def __init__(self, source_code: str, source_lines: Sequence[str], file_path: str) -> None:
        self.source_code = source_code
        self.source_lines = source_lines
        self.file_path = file_path
        self.symbols: List[CodeSymbol] = []
        self.file_imports: List[str] = []
        self._current_class: Optional[str] = None

    def _extract_source_segment(self, start_line: int, end_line: int) -> str:
        """Extract lines of code between 1-indexed line numbers inclusive."""
        if start_line < 1 or end_line < start_line or start_line > len(self.source_lines):
            return ""
        return "\n".join(self.source_lines[start_line - 1 : min(end_line, len(self.source_lines))])

    def visit_Import(self, node: ast.Import) -> None:
        """Extract standard imports (e.g., import os, import numpy as np)."""
        for alias in node.names:
            import_str = f"import {alias.name}"
            if alias.asname:
                import_str += f" as {alias.asname}"
            self.file_imports.append(import_str)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract from imports (e.g., from math import sqrt)."""
        module = node.module or ""
        prefix = "." * node.level + module
        for alias in node.names:
            import_str = f"from {prefix} import {alias.name}"
            if alias.asname:
                import_str += f" as {alias.asname}"
            self.file_imports.append(import_str)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function or method definitions."""
        self._handle_function_node(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract asynchronous function or method definitions."""
        self._handle_function_node(node, is_async=True)

    def _handle_function_node(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_async: bool = False,
    ) -> None:
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)

        code_segment = self._extract_source_segment(start_line, end_line)

        if self._current_class:
            symbol_name = f"{self._current_class}.{node.name}"
            symbol_type = "method"
        else:
            symbol_name = node.name
            symbol_type = "function"

        symbol = CodeSymbol(
            name=symbol_name,
            symbol_type=symbol_type,
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            code=code_segment,
            imports=list(self.file_imports),
        )
        self.symbols.append(symbol)

        # Do not recurse into nested function bodies to avoid duplicate fragments,
        # but if needed, generic_visit can be called. Here, top-level functions and
        # class methods provide clean self-contained units.

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions and then visit methods inside the class."""
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", start_line)
        code_segment = self._extract_source_segment(start_line, end_line)

        class_symbol = CodeSymbol(
            name=node.name,
            symbol_type="class",
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            code=code_segment,
            imports=list(self.file_imports),
        )
        self.symbols.append(class_symbol)

        previous_class = self._current_class
        self._current_class = node.name
        # Visit inner nodes (methods)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)
            elif isinstance(item, ast.ClassDef):
                self.visit(item)
        self._current_class = previous_class


def safe_extract_zip(
    zip_source: Union[str, Path, bytes, io.BytesIO],
    target_dir: Union[str, Path],
) -> Path:
    """
    Safely unpack a ZIP archive to target directory, guarding against ZipSlip.

    Args:
        zip_source: Path to zip file, or raw bytes / BytesIO buffer.
        target_dir: Destination directory path.

    Returns:
        Path to unpacked directory.

    Raises:
        ValueError: If a zip member contains dangerous path traversal components.
        zipfile.BadZipFile: If source is not a valid zip archive.
    """
    dest_path = Path(target_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    if isinstance(zip_source, (str, Path)):
        archive_context = zipfile.ZipFile(zip_source, "r")
    elif isinstance(zip_source, bytes):
        archive_context = zipfile.ZipFile(io.BytesIO(zip_source), "r")
    elif isinstance(zip_source, io.BytesIO):
        archive_context = zipfile.ZipFile(zip_source, "r")
    else:
        raise TypeError(f"Unsupported zip source type: {type(zip_source)}")

    with archive_context as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            # Guard against ZipSlip path traversal
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP member path detected: {member.filename}")

            resolved_path = (dest_path / member_path).resolve()
            if not str(resolved_path).startswith(str(dest_path)):
                raise ValueError(f"Target path escape detected: {member.filename}")

            archive.extract(member, dest_path)

    return dest_path


class RepositoryParser:
    """Scanner and AST parser for repository codebases."""

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
        ignored_directories: Optional[Set[str]] = None,
    ) -> None:
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        self.ignored_directories = ignored_directories or IGNORED_DIRECTORIES

    def is_ignored(self, path: Path, base_dir: Path) -> bool:
        """Check whether any component of the relative path is in ignored directories."""
        try:
            rel_parts = path.relative_to(base_dir).parts
        except ValueError:
            rel_parts = path.parts

        for part in rel_parts:
            if part in self.ignored_directories:
                return True
            if part.startswith(".") and part not in {".", ".."}:
                # Ignore hidden directories like .cache, .git, etc.
                return True
        return False

    def parse_python_file(self, file_path: Path, base_dir: Path) -> List[CodeSymbol]:
        """Parse a Python file using the AST module."""
        symbols: List[CodeSymbol] = []
        try:
            rel_path = str(file_path.relative_to(base_dir)).replace("\\", "/")
        except ValueError:
            rel_path = str(file_path).replace("\\", "/")

        try:
            source_code = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as err:
            logger.warning("Failed to read file %s: %s", file_path, err)
            return symbols

        source_lines = source_code.splitlines()
        total_lines = len(source_lines)
        if total_lines == 0:
            return symbols

        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as err:
            logger.warning("SyntaxError in %s: %s. Falling back to whole-file symbol.", file_path, err)
            return [
                CodeSymbol(
                    name=file_path.stem,
                    symbol_type="module",
                    file_path=rel_path,
                    start_line=1,
                    end_line=total_lines,
                    code=source_code,
                    imports=[],
                )
            ]

        # Extract file-level imports and docstring
        file_docstring = ast.get_docstring(tree) or ""
        visitor = PythonASTVisitor(source_code, source_lines, rel_path)
        visitor.visit(tree)

        # Module-level symbol to represent file context and top-level docstring/overview
        module_symbol = CodeSymbol(
            name=file_path.stem,
            symbol_type="module",
            file_path=rel_path,
            start_line=1,
            end_line=total_lines,
            code=f'"""{file_docstring}"""\n' + "\n".join(visitor.file_imports) if file_docstring else "\n".join(visitor.file_imports),
            imports=list(visitor.file_imports),
        )
        symbols.append(module_symbol)
        symbols.extend(visitor.symbols)

        return symbols

    def parse_text_file(self, file_path: Path, base_dir: Path) -> List[CodeSymbol]:
        """Parse non-Python code or markdown file with line boundaries."""
        symbols: List[CodeSymbol] = []
        try:
            rel_path = str(file_path.relative_to(base_dir)).replace("\\", "/")
        except ValueError:
            rel_path = str(file_path).replace("\\", "/")

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as err:
            logger.warning("Failed to read non-python file %s: %s", file_path, err)
            return symbols

        lines = content.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return symbols

        symbol_type = "doc" if file_path.suffix.lower() == ".md" else "code"
        symbols.append(
            CodeSymbol(
                name=file_path.name,
                symbol_type=symbol_type,
                file_path=rel_path,
                start_line=1,
                end_line=total_lines,
                code=content,
                imports=[],
            )
        )
        return symbols

    def scan_directory(self, target_dir: Union[str, Path]) -> List[CodeSymbol]:
        """
        Recursively scan a directory for supported files and extract symbols.

        Args:
            target_dir: Root directory path of codebase.

        Returns:
            List of CodeSymbol objects.
        """
        base_path = Path(target_dir).resolve()
        if not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

        all_symbols: List[CodeSymbol] = []

        for root, dirs, files in os.walk(base_path):
            current_dir = Path(root)
            # Prune ignored directories in-place
            dirs[:] = [
                d
                for d in dirs
                if not self.is_ignored(current_dir / d, base_path)
            ]

            for file_name in files:
                file_path = current_dir / file_name
                if self.is_ignored(file_path, base_path):
                    continue

                suffix = file_path.suffix.lower()
                if suffix not in self.supported_extensions:
                    continue

                if suffix == ".py":
                    symbols = self.parse_python_file(file_path, base_path)
                else:
                    symbols = self.parse_text_file(file_path, base_path)

                all_symbols.extend(symbols)

        return all_symbols


def parse_repository(
    source: Union[str, Path, bytes, io.BytesIO],
    extract_target_dir: Optional[Union[str, Path]] = None,
) -> List[CodeSymbol]:
    """
    Convenience function to parse a repository from a directory path or ZIP archive.

    Args:
        source: Directory path, zip archive path, or zip bytes.
        extract_target_dir: Optional extraction target for zip archives.

    Returns:
        List of parsed CodeSymbol objects.
    """
    parser = RepositoryParser()

    # If source is bytes or BytesIO, unpack zip
    if isinstance(source, (bytes, io.BytesIO)):
        target = extract_target_dir or Path(".gitrag_extracted")
        unpacked_dir = safe_extract_zip(source, target)
        return parser.scan_directory(unpacked_dir)

    source_path = Path(source)
    if source_path.is_file() and zipfile.is_zipfile(source_path):
        target = extract_target_dir or (source_path.parent / f"{source_path.stem}_unpacked")
        unpacked_dir = safe_extract_zip(source_path, target)
        return parser.scan_directory(unpacked_dir)

    if source_path.is_dir():
        return parser.scan_directory(source_path)

    raise ValueError(f"Source must be a directory or ZIP file: {source}")
