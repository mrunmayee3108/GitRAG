"""
Vector storage and semantic retrieval engine for GitRAG.

Utilizes Sentence Transformers (all-MiniLM-L6-v2) to generate dense embeddings
and FAISS (CPU) for fast vector indexing and cosine-similarity retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_STORAGE_DIR = "storage/faiss_index"


class VectorStore:
    """FAISS-based vector store and query engine."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Initialize the VectorStore.

        Args:
            model_name: HuggingFace model identifier for SentenceTransformer.
        """
        self.model_name = model_name
        self._model: Optional[Any] = None

    @property
    def model(self) -> Any:
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it using: pip install sentence-transformers"
                ) from err
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate dense embeddings for a list of text strings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size during encoding.
            normalize: Whether to L2-normalize vectors for cosine similarity.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings.astype(np.float32)

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Build a FAISS vector index from chunk dictionaries.

        Args:
            chunks: List of chunk dicts containing 'text' and 'metadata'.
            save_dir: Optional directory to persist index and metadata.

        Returns:
            faiss.Index instance.
        """
        try:
            import faiss
        except ImportError as err:
            raise ImportError(
                "faiss-cpu is not installed. Please install it using: pip install faiss-cpu"
            ) from err

        if not chunks:
            logger.warning("Empty chunks provided to build_index. Creating empty index.")
            dim = 384  # default all-MiniLM-L6-v2 dimension
            index = faiss.IndexFlatIP(dim)
            if save_dir:
                self.save(save_dir, index, chunks)
            return index

        texts = [chunk.get("text", "") for chunk in chunks]
        embeddings = self.encode(texts, normalize=True)
        dimension = embeddings.shape[1]

        # Inner Product with normalized embeddings equals Cosine Similarity
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        logger.info("Built FAISS index with %d vectors (dimension %d)", index.ntotal, dimension)

        if save_dir:
            self.save(save_dir, index, chunks)

        return index

    def save(
        self,
        save_dir: Union[str, Path],
        index: Any,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Persist FAISS index and chunk metadata to disk.

        Args:
            save_dir: Directory to save artifacts.
            index: FAISS index instance.
            chunks: Associated list of chunk dictionaries.
        """
        try:
            import faiss
        except ImportError as err:
            raise ImportError("faiss-cpu is required to save index") from err

        target_path = Path(save_dir).resolve()
        target_path.mkdir(parents=True, exist_ok=True)

        index_file = target_path / "index.faiss"
        chunks_file = target_path / "chunks.json"

        faiss.write_index(index, str(index_file))
        chunks_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        logger.info("Saved FAISS index and metadata to: %s", target_path)

    def load(
        self,
        save_dir: Union[str, Path],
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Load persisted FAISS index and chunk metadata from disk.

        Args:
            save_dir: Directory where index.faiss and chunks.json reside.

        Returns:
            Tuple of (faiss.Index, list of chunk dictionaries).
        """
        try:
            import faiss
        except ImportError as err:
            raise ImportError("faiss-cpu is required to load index") from err

        target_path = Path(save_dir).resolve()
        index_file = target_path / "index.faiss"
        chunks_file = target_path / "chunks.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {index_file}")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks metadata file not found at: {chunks_file}")

        index = faiss.read_index(str(index_file))
        chunks: List[Dict[str, Any]] = json.loads(chunks_file.read_text(encoding="utf-8"))
        logger.info("Loaded FAISS index (%d items) from %s", index.ntotal, target_path)
        return index, chunks

    def query(
        self,
        query_text: str,
        index: Any,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query codebase index using semantic search.

        Args:
            query_text: Natural language or code search query.
            index: Built FAISS index.
            chunks: List of chunk dicts aligned with index vectors.
            top_k: Number of highest matching snippets to retrieve.

        Returns:
            List of result dicts with 'score', 'text', and 'metadata'.
        """
        if index.ntotal == 0 or not chunks:
            return []

        query_vec = self.encode([query_text], normalize=True)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                }
            )

        return results


# Module-level singleton instance for convenience
_default_store: Optional[VectorStore] = None


def _get_store(model_name: str = DEFAULT_MODEL_NAME) -> VectorStore:
    """Retrieve or create a VectorStore instance."""
    global _default_store
    if _default_store is None or _default_store.model_name != model_name:
        _default_store = VectorStore(model_name=model_name)
    return _default_store


def build_repo_index(
    chunks: List[Dict[str, Any]],
    save_dir: str = DEFAULT_STORAGE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Any:
    """
    Build and persist a FAISS index from codebase chunks.

    Args:
        chunks: List of chunk dictionaries.
        save_dir: Directory to save index.faiss and chunks.json.
        model_name: Model identifier for embeddings.

    Returns:
        faiss.Index: The built FAISS index.
    """
    store = _get_store(model_name=model_name)
    return store.build_index(chunks=chunks, save_dir=save_dir)


def load_repo_index(
    save_dir: str = DEFAULT_STORAGE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Load an existing FAISS index and chunk metadata from disk.

    Args:
        save_dir: Directory containing index.faiss and chunks.json.
        model_name: Model identifier for embeddings.

    Returns:
        Tuple of (faiss.Index, list of chunks).
    """
    store = _get_store(model_name=model_name)
    return store.load(save_dir=save_dir)


def query_codebase(
    query: str,
    index: Any,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """
    Query the codebase index using dense retrieval.

    Args:
        query: Search query string.
        index: FAISS index.
        chunks: Aligned list of chunk dictionaries.
        top_k: Max count of top matches to retrieve.
        model_name: Embedding model identifier.

    Returns:
        List of matching chunks with similarity scores and metadata.
    """
    store = _get_store(model_name=model_name)
    return store.query(query_text=query, index=index, chunks=chunks, top_k=top_k)
