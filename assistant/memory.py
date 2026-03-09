"""Memory management with optional semantic retrieval.

This module provides a simple SQLite-backed memory database as well as a
light‑weight FAISS/`all-MiniLM-L6-v2` vector index for semantic search.
"""

import datetime
import sqlite3
import pickle
from typing import Optional, List, Dict
import functools

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# --- configuration --------------------------------------------------------
DB_PATH = 'assistant_memory.db'  # sqlite file storing raw memories

# --- internal globals -----------------------------------------------------
_id_map: List[int] = []  # maps faiss index position → memory id
_index: Optional[faiss.Index] = None  # FAISS index instance, created lazily
_model: Optional[SentenceTransformer] = None  # embedding model loaded on demand


def _get_model() -> SentenceTransformer:
    """Return a shared embedding model, loading it if necessary.

    The `device` argument respects whether a CUDA-capable GPU is available
    in the current PyTorch build.  If CUDA is not enabled, fall back to CPU to
    avoid runtime assertions.
    """
    global _model
    if _model is None:
        # PyTorch may not be compiled with CUDA support in this environment.
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _model


# ---------------------------------------------------------------------------
# Core database operations
# ---------------------------------------------------------------------------

def init_memory_db():
    """Create the SQLite database if it doesn't exist and rebuild vector index."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS memories
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  user_input TEXT,
                  ai_response TEXT,
                  summary TEXT)''')
    conn.commit()
    conn.close()

    # rebuild semantic index to keep it in sync with the database
    rebuild_index()


def save_memory(user_input: str, ai_response: str):
    """Persist a memory and update semantic index.

    The summary is truncated for storage so the embedding remains small.
    """
    summary = f"User: {user_input[:50]}... AI: {ai_response[:50]}..."
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memories (timestamp, user_input, ai_response, summary) VALUES (?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), user_input, ai_response, summary),
    )
    conn.commit()
    mem_id: Optional[int] = c.lastrowid
    conn.close()

    # update semantic index with the new summary
    if mem_id is not None:
        _add_to_index(mem_id, summary)


def get_recent_memories(limit: int = 5) -> list[str]:
    """Return the most recent memory summaries (chronological order)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT summary FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    # reverse to chronological order
    return [row[0] for row in rows[::-1]]


# ---------------------------------------------------------------------------
# Semantic retrieval utilities
# ---------------------------------------------------------------------------

def _ensure_index(dim: int = 384):
    """Ensure that a FAISS index object exists in ``_index``.

    Attempt to load a previously persisted index and ID map from disk; if that
    fails (e.g. first run or corrupted file), create a new flat L2 index with
    the given dimensionality.
    """
    global _index
    if _index is None:
        try:
            # try loading persisted index
            _index = faiss.read_index('assistant_memory.index')
            # load _id_map from file if exists
            with open('assistant_memory.map', 'rb') as f:
                data = pickle.load(f)
                _id_map.clear()
                _id_map.extend(data)
        except Exception:
            _index = faiss.IndexFlatL2(dim)


@functools.lru_cache(maxsize=1024)
def _embed_text(text: str) -> np.ndarray:
    """Return an embedding vector for the given text.

    Results are cached in-process so that repeated calls with the same
    string return quickly without invoking the model again.
    """
    # note: caching decorator uses the text string as key
    model = _get_model()
    # sentence-transformers returns numpy array
    return model.encode(text, convert_to_numpy=True)


def _add_to_index(mem_id: int, summary: str):
    """Embed a summary and add it to the FAISS index.

    The corresponding ``mem_id`` is appended to ``_id_map`` so the original
    SQLite row can be looked up during searches. The index/map are persisted
    after modification.
    """
    emb = _embed_text(summary).astype('float32')
    _ensure_index()
    assert _index is not None, "index should be initialized"
    _index.add(emb.reshape(1, -1))  # type: ignore[arg-type]
    _id_map.append(mem_id)
    # persist
    try:
        faiss.write_index(_index, 'assistant_memory.index')
        with open('assistant_memory.map', 'wb') as f:
            pickle.dump(_id_map, f)
    except Exception:
        pass


def rebuild_index():
    """Rebuild the FAISS index from all entries stored in the database.

    This drops any existing in-memory index and repopulates it from scratch,
    also persisting the resulting index and ID map to disk.
    """
    global _id_map, _index
    _id_map = []
    _ensure_index()
    assert _index is not None, "index should be initialized"
    _index.reset()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, summary FROM memories")
    for mem_id, summary in c.fetchall():
        emb = _embed_text(summary).astype('float32')
        assert _index is not None, "index should be initialized"
        _index.add(emb.reshape(1, -1))  # type: ignore[arg-type]
        _id_map.append(mem_id)
    conn.close()
    # persist index and map
    try:
        faiss.write_index(_index, 'assistant_memory.index')
        with open('assistant_memory.map', 'wb') as f:
            pickle.dump(_id_map, f)
    except Exception:
        pass


def search_semantic(query: str, top_k: int = 5) -> list[str]:
    """Return the most-relevant memory summaries for ``query``.

    The function returns up to ``top_k`` summaries ranked by cosine/L2
    distance in embedding space. If the index is empty no results are
    returned.
    """
    emb = _embed_text(query).astype('float32')
    _ensure_index()
    assert _index is not None, "index should be initialized"
    if _index.ntotal == 0:
        return []
    _, I = _index.search(emb.reshape(1, -1), top_k)  # type: ignore[arg-type]
    results: list[str] = []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for idx in I[0]:
        if idx < 0 or idx >= len(_id_map):
            continue
        mem_id = _id_map[idx]
        c.execute("SELECT summary FROM memories WHERE id=?", (mem_id,))
        row = c.fetchone()
        if row:
            results.append(row[0])
    conn.close()
    return results


# public API
__all__ = [
    "init_memory_db",
    "save_memory",
    "get_recent_memories",
    "search_semantic",
]
