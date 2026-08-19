"""
Vector memory for Jarvis — semantic recall instead of keyword grep.

Uses ChromaDB when available (``pip install chromadb``); otherwise falls back
to a fully-offline, dependency-free in-process store with a lightweight
bag-of-words/char-ngram embedder.  The public API is identical either way, so
callers don't care which backend is active.

Wire-in point: replace the keyword scan in ``memory/memory_manager.py`` with
``VectorMemory.query(text)`` to get the most relevant past facts/notes.

Example::

    from core.vector_memory import VectorMemory
    vm = VectorMemory()
    vm.add("user likes dark mode", {"kind": "preference"})
    vm.add("user is learning rust", {"kind": "project"})
    print(vm.query("what does the user prefer?"))   # -> ranked results
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

from core.security import get_base_dir

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

# Stopwords dilute the offline bag-of-ngrams signal (e.g. "the cat sat on the
# mat" would otherwise rank above a genuinely relevant doc).  Filtered out so
# recall is meaningful without a transformer model.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "at", "by", "from", "about", "into", "than",
    "then", "so", "do", "does", "did", "does", "you", "your", "i", "we", "they",
    "he", "she", "what", "which", "who", "how", "when", "where", "why", "not",
    "no", "yes", "can", "will", "would", "should", "could", "may", "might",
    "does", "has", "have", "had", "my", "me", "our", "us", "them", "their",
    "his", "her", "if", "else", "while", "because", "there", "here", "out",
}


def _tokens(text: str) -> list[str]:
    return [
        t for t in _TOKEN.findall((text or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


class _Vec:
    """Sparse vector helper (dict of key->weight)."""

    __slots__ = ("d",)

    def __init__(self, d: dict[str, float]) -> None:
        self.d = d

    def dot(self, other: "_Vec") -> float:
        if len(self.d) <= len(other.d):
            return sum(w * other.d.get(k, 0.0) for k, w in self.d.items())
        return sum(w * self.d.get(k, 0.0) for k, w in other.d.items())

    def norm(self) -> float:
        return math.sqrt(sum(w * w for w in self.d.values())) or 1.0

    def cos(self, other: "_Vec") -> float:
        return self.dot(other) / (self.norm() * other.norm())


def _embed(text: str) -> _Vec:
    text = (text or "").lower()
    d: dict[str, float] = {}
    for tok in _tokens(text):
        d[tok] = d.get(tok, 0.0) + 1.0
    for i in range(len(text) - 2):
        ngram = text[i : i + 3]
        if ngram.strip():
            d["#" + ngram] = d.get("#" + ngram, 0.0) + 0.5
    return _Vec(d)


class VectorMemory:
    def __init__(self, name: str = "jarvis", persist_dir: str | Path | None = None) -> None:
        self.name = name
        self.persist_dir = Path(persist_dir) if persist_dir else (get_base_dir() / "memory" / "vectors")
        self._lock = __import__("threading").Lock()
        self._docs: list[dict[str, Any]] = []
        self._client = None
        self._collection = None
        self._backend = "offline"
        self._try_chroma()

    def _try_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(name=self.name)
            self._backend = "chromadb"
            logger.info("VectorMemory backend: chromadb (%s)", self.persist_dir)
        except Exception as exc:  # noqa: BLE001 - chromadb optional
            logger.info("VectorMemory backend: offline embedder (chromadb unavailable: %s)", exc)

    # ── public API ────────────────────────────────────────────────────────────
    def add(self, text: str, metadata: dict[str, Any] | None = None, id: str | None = None) -> str:
        if self._backend == "chromadb":
            cid = id or f"doc_{len(self._docs)}_{abs(hash(text))}"
            self._collection.add(documents=[text], metadatas=[metadata], ids=[cid])  # type: ignore[union-attr]
            return cid
        with self._lock:
            cid = id or f"doc_{len(self._docs)}_{abs(hash(text))}"
            self._docs.append({"id": cid, "text": text, "meta": metadata or {}, "vec": _embed(text)})
        return cid

    def query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._backend == "chromadb":
            try:
                res = self._collection.query(query_texts=[text], n_results=top_k)  # type: ignore[union-attr]
                out = []
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for d, m, dist in zip(docs, metas, dists):
                    out.append({"text": d, "metadata": m, "score": 1.0 - float(dist)})
                return out
            except Exception as exc:  # noqa: BLE001 - fall back to offline if chroma query fails
                logger.warning("chromadb query failed, using offline: %s", exc)
        q = _embed(text)
        with self._lock:
            scored = [(d["vec"].cos(q), d) for d in self._docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": d["text"], "metadata": d["meta"], "score": float(s)}
            for s, d in scored[:top_k]
            if s > 0.0
        ]

    def count(self) -> int:
        if self._backend == "chromadb":
            try:
                return int(self._collection.count())  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                return 0
        return len(self._docs)

    def clear(self) -> None:
        if self._backend == "chromadb":
            try:
                self._client.delete_collection(self.name)  # type: ignore[union-attr]
                self._collection = self._client.get_or_create_collection(name=self.name)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        else:
            with self._lock:
                self._docs.clear()
