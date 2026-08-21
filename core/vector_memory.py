"""
Vector memory for Jarvis — semantic recall instead of keyword grep.

Uses ChromaDB when available (``pip install chromadb``); otherwise falls back
to a fully-offline, dependency-free in-process store with a lightweight
bag-of-words/char-ngram embedder.  The public API is identical either way, so
callers don't care which backend is active.

JARVIS 7.0 enhancements:
  * **Memory decay** — importance decays exponentially for episodic memories
    (half-life of N days), but procedural memories are preserved.
  * **Importance scoring** — frequently-accessed memories rank higher.
  * **Separation** — episodic (what happened), semantic (facts/preferences),
    procedural (learned lessons about how to do tasks).

Wire-in point: replace the keyword scan in ``memory/memory_manager.py`` with
``VectorMemory.query(text)`` to get the most relevant past facts/notes.

Example::

    from core.vector_memory import VectorMemory
    vm = VectorMemory()
    vm.add("user likes dark mode", {"kind": "preference"}, mem_type="semantic")
    vm.add("user is learning rust", {"kind": "project"}, mem_type="semantic")
    vm.add("met Sarah at conference yesterday", mem_type="episodic")
    vm.add("always check git status before committing", mem_type="procedural")
    print(vm.query("what does the user prefer?"))   # -> ranked results
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
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
    "has", "have", "had", "my", "me", "our", "us", "them", "their",
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


def _get_embedding(text: str) -> list[float]:
    """Get a dense embedding from the local embedder or model.

    Tries to use a lightweight local embedding model if available;
    otherwise returns a sparse vector from the offline embedder.
    """
    try:
        import numpy as np  # noqa
        # Try sentence-transformers if installed
        from sentence_transformers import SentenceTransformer
        model = getattr(_get_embedding, "_model", None)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _get_embedding._model = model
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    except ImportError:
        pass
    # Fallback: use offline sparse embedder as a dense-ish vector
    v = _embed(text)
    # Convert sparse dict to fixed-size vector (top 100 tokens by weight)
    sorted_items = sorted(v.d.items(), key=lambda x: x[1], reverse=True)[:100]
    return [w for _, w in sorted_items]


class VectorMemory:
    """Thread-safe vector memory with importance-based recall and decay.

    Memory types:
      * ``episodic``   — personal events / conversations (decays over time)
      * ``semantic``   — facts, preferences, world knowledge (stable)
      * ``procedural`` — how-to knowledge, lessons (never decays)
    """

    def __init__(self, name: str = "jarvis", persist_dir: str | Path | None = None,
                 half_life_days: float = 30.0) -> None:
        self.name = name
        self.persist_dir = Path(persist_dir) if persist_dir else (get_base_dir() / "memory" / "vectors")
        self._lock = threading.Lock()
        self._half_life_days = half_life_days
        self._docs: list[dict[str, Any]] = []
        self._client = None
        self._collection = None
        self._backend = "offline"
        self._try_chroma()
        self._load_persistent()

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

    def _load_persistent(self) -> None:
        """Load in-memory docs from a local JSON cache (offline backend only)."""
        cache_path = self.persist_dir / f"{self.name}_cache.json"
        if self._backend == "offline" and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                self._docs = data
                logger.info("Loaded %d cached memories", len(self._docs))
            except Exception:
                pass

    def _save_persistent(self) -> None:
        """Persist in-memory docs to a local JSON cache."""
        if self._backend != "offline":
            return
        cache_path = self.persist_dir / f"{self.name}_cache.json"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(self._docs, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to cache vector memory: %s", exc)

    # ── public API ────────────────────────────────────────────────────────────
    def add(self, text: str, metadata: dict[str, Any] | None = None,
            id: str | None = None, mem_type: str = "semantic",
            importance: float = 0.5) -> str:
        """Add a memory with type classification and importance score."""
        cid = id or f"doc_{uuid.uuid4().hex}"
        meta = {"mem_type": mem_type, "importance": importance,
                **(metadata or {})}
        now = time.time()

        if self._backend == "chromadb":
            self._collection.add(  # type: ignore
                documents=[text],
                metadatas=[meta],
                embeddings=[_get_embedding(text)],
                ids=[cid],
            )
            return cid

        doc = {
            "id": cid, "text": text, "meta": meta,
            "vec": _embed(text),  # sparse vector for offline search
            "dense_vec": _get_embedding(text),  # dense vector for potential use
            "created_at": now,
            "accessed_at": now,
            "access_count": 0,
        }
        with self._lock:
            self._docs.append(doc)
            self._save_persistent()
        return cid

    def query(self, text: str, top_k: int = 5,
              mem_type: str | None = None) -> list[dict[str, Any]]:
        """Search memories by semantic similarity, with importance re-ranking."""
        if self._backend == "chromadb":
            return self._query_chroma(text, top_k, mem_type)
        return self._query_offline(text, top_k, mem_type)

    def _query_chroma(self, text: str, top_k: int, mem_type: str | None) -> list[dict[str, Any]]:
        try:
            where = {"mem_type": mem_type} if mem_type else None
            res = self._collection.query(  # type: ignore
                query_texts=[text], n_results=top_k * 3,
                where=where if where else None,
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]

            scored: list[tuple[float, dict]] = []
            for d, m, dist in zip(docs, metas, dists):
                sim = max(0.0, 1.0 - float(dist) / 2.0)  # normalize to [0,1]
                imp = float(m.get("importance", 0.5))
                # Decay episodic memories
                if m.get("mem_type") == "episodic":
                    created = m.get("created_at", time.time())
                    age_days = (time.time() - created) / 86400
                    decay = math.pow(0.5, age_days / self._half_life_days)
                    imp *= decay
                final_score = sim * (0.5 + imp)  # importance boosts relevance
                scored.append((final_score, {
                    "text": d, "metadata": m,
                    "score": float(sim), "final_score": float(final_score),
                }))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s[1] for s in scored[:top_k] if s[0] > 0.0]
        except Exception as exc:
            logger.warning("chromadb query failed, using offline: %s", exc)
            return self._query_offline(text, top_k, mem_type)

    def _query_offline(self, text: str, top_k: int, mem_type: str | None) -> list[dict[str, Any]]:
        q = _embed(text)
        with self._lock:
            candidates = [d for d in self._docs
                         if mem_type is None or d["meta"].get("mem_type") == mem_type]
            scored = [(d["vec"].cos(q), d) for d in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for s, d in scored[:top_k]:
            if s <= 0.0:
                continue
            meta = d["meta"]
            now = time.time()
            created = d.get("created_at", now)
            access_count = d.get("access_count", 0) + 1
            d["accessed_at"] = now
            d["access_count"] = access_count

            # Decay episodic memories
            if meta.get("mem_type") == "episodic":
                age_days = (now - created) / 86400
                decay = math.pow(0.5, age_days / self._half_life_days)
            else:
                decay = 1.0

            # Importance boost from access count (frequently referenced = important)
            access_bonus = min(access_count / 10.0, 0.5)
            importance = float(meta.get("importance", 0.5)) * decay + access_bonus
            final_score = float(s) * (0.5 + importance)

            results.append({
                "text": d["text"], "metadata": meta,
                "score": float(s), "final_score": float(final_score),
                "access_count": access_count,
            })
        with self._lock:
            self._save_persistent()
        return results

    def count(self, mem_type: str | None = None) -> int:
        if self._backend == "chromadb":
            try:
                if mem_type:
                    result = self._collection.get(where={"mem_type": mem_type}, limit=1)
                    return len(result.get("ids", [])) if result else 0
                return int(self._collection.count())
            except Exception:
                return 0
        with self._lock:
            if mem_type:
                return sum(1 for d in self._docs if d["meta"].get("mem_type") == mem_type)
            return len(self._docs)

    def decay(self, half_life_days: float | None = None) -> int:
        """Apply memory decay to episodic memories and return count of decayed."""
        half_life = half_life_days or self._half_life_days
        decayed = 0
        with self._lock:
            for d in self._docs:
                meta = d["meta"]
                if meta.get("mem_type") != "episodic":
                    continue
                created = d.get("created_at", time.time())
                age_days = (time.time() - created) / 86400
                decay_factor = math.pow(0.5, age_days / half_life)
                old_imp = float(meta.get("importance", 0.5))
                new_imp = old_imp * decay_factor
                if new_imp < 0.05 and old_imp >= 0.05:
                    # Memory is now stale — remove it
                    self._docs.remove(d)
                    decayed += 1
                else:
                    meta["importance"] = new_imp
        if decayed:
            self._save_persistent()
        return decayed

    def forget(self, doc_id: str) -> bool:
        if self._backend == "chromadb":
            try:
                self._collection.delete(ids=[doc_id])
                return True
            except Exception:
                return False
        with self._lock:
            self._docs = [d for d in self._docs if d["id"] != doc_id]
            self._save_persistent()
            return True

    def clear(self) -> None:
        if self._backend == "chromadb":
            try:
                self._client.delete_collection(self.name)
                self._collection = self._client.get_or_create_collection(name=self.name)
            except Exception:
                pass
        else:
            with self._lock:
                self._docs.clear()
            cache_path = self.persist_dir / f"{self.name}_cache.json"
            cache_path.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        """Return counts by memory type."""
        if self._backend == "chromadb":
            return {
                "total": self.count(),
                "episodic": self.count("episodic"),
                "semantic": self.count("semantic"),
                "procedural": self.count("procedural"),
            }
        with self._lock:
            return {
                "total": len(self._docs),
                "episodic": sum(1 for d in self._docs if d["meta"].get("mem_type") == "episodic"),
                "semantic": sum(1 for d in self._docs if d["meta"].get("mem_type") == "semantic"),
                "procedural": sum(1 for d in self._docs if d["meta"].get("mem_type") == "procedural"),
            }


# Process-wide instance.
vector_db = VectorMemory()
