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

Two embedders are used, and it matters which one is active:

  * ``all-MiniLM-L6-v2`` via ``sentence-transformers`` when installed and
    loadable (384-dim, L2-normalised).
  * Otherwise a **fixed-dimension signed-hash projection** of the same sparse
    features the offline search uses, also 384-dim and L2-normalised.

Both produce 384 dimensions on purpose: chromadb pins a collection to the
dimensionality of its first insert, so a ragged fallback would make every later
add fail.  They are *not* the same vector space though, so a store written with
one and queried with the other recalls noise.  The active embedder is recorded
on each memory's metadata as ``embedder`` and logged once at startup, so a
mid-life switch is diagnosable rather than mysterious.

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

import logging
import math
import re
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

from core.json_store import atomic_write_json, read_json
from core.security import get_base_dir

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

#: Dimensionality of every dense vector this module produces.  Matches
#: ``all-MiniLM-L6-v2`` so the two embedders stay interchangeable as far as
#: chromadb's fixed-width collections are concerned.
_DENSE_DIM = 384

#: Bumped when the on-disk cache layout changes incompatibly.
_SCHEMA_VERSION = 2

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


# ── dense embedding ───────────────────────────────────────────────────────────
# ``sentence-transformers`` is optional and its constructor reaches the network
# to validate the HuggingFace cache.  A failure there is not an ImportError, so
# it must be caught broadly -- and remembered, or every single add() retries a
# network round-trip.
_embedder_lock = threading.Lock()
_embedder_model: Any = None
_embedder_failed = False


def _hashed_dense(text: str) -> list[float]:
    """Project the sparse features into a fixed ``_DENSE_DIM`` vector.

    Signed feature hashing: each feature lands in one bucket with a sign taken
    from the same hash, so collisions tend to cancel instead of accumulating.
    ``zlib.crc32`` is used rather than ``hash()`` because ``hash()`` of a str is
    salted per process (PYTHONHASHSEED) -- vectors written in one run would not
    match vectors written in the next.
    """
    vec = [0.0] * _DENSE_DIM
    for key, weight in _embed(text).d.items():
        h = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[h % _DENSE_DIM] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [v / norm for v in vec]
    return vec


def _load_dense_model() -> Any:
    """Return the sentence-transformers model, or None if unavailable.

    The negative result is cached: an offline machine degrades once, quietly.
    """
    global _embedder_model, _embedder_failed
    if _embedder_model is not None:
        return _embedder_model
    if _embedder_failed:
        return None
    with _embedder_lock:
        if _embedder_model is not None:
            return _embedder_model
        if _embedder_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            _embedder_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("VectorMemory embedder: all-MiniLM-L6-v2 (%d-dim)", _DENSE_DIM)
        except Exception as exc:  # noqa: BLE001 - optional dep, may fail at load
            _embedder_failed = True
            logger.info(
                "VectorMemory embedder: hashed fallback (%d-dim); "
                "sentence-transformers unavailable (%s: %s)",
                _DENSE_DIM, type(exc).__name__, exc,
            )
        return _embedder_model


def embedder_name() -> str:
    """Name of the embedder that will be used for the next call."""
    return "minilm" if _load_dense_model() is not None else "hashed"


def _get_embedding(text: str) -> list[float]:
    """Return an L2-normalised ``_DENSE_DIM``-dimensional embedding.

    Always returns exactly ``_DENSE_DIM`` floats.  Never raises: a dense-model
    failure degrades to the hashed projection instead of propagating out of
    ``add()`` and losing the memory.
    """
    model = _load_dense_model()
    if model is not None:
        try:
            emb = model.encode(text or "", normalize_embeddings=True)
            dense = [float(x) for x in emb.tolist()]
            if len(dense) == _DENSE_DIM:
                return dense
            logger.warning(
                "dense embedder returned %d dims, expected %d; using hashed fallback",
                len(dense), _DENSE_DIM,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never lose the memory
            logger.warning("dense encode failed (%s); using hashed fallback: %s",
                           type(exc).__name__, exc)
    return _hashed_dense(text)


# ── on-disk representation ────────────────────────────────────────────────────
def _doc_to_json(doc: dict[str, Any]) -> dict[str, Any]:
    """Serialise a doc, unwrapping the ``_Vec`` that json cannot encode."""
    return {
        "id": doc["id"],
        "text": doc["text"],
        "meta": doc["meta"],
        "vec": doc["vec"].d,
        "created_at": doc["created_at"],
        "accessed_at": doc["accessed_at"],
        "access_count": doc["access_count"],
    }


def _doc_from_json(raw: Any) -> dict[str, Any] | None:
    """Rehydrate a doc from JSON, or return None if it is unusable.

    ``text`` is the only irreplaceable field -- a missing or corrupt ``vec`` is
    simply recomputed from it rather than costing us the memory.
    """
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    stored = raw.get("vec")
    vec: _Vec | None = None
    if isinstance(stored, dict):
        weights = {
            k: float(v) for k, v in stored.items()
            if isinstance(k, str) and isinstance(v, (int, float))
        }
        if weights:
            vec = _Vec(weights)
    if vec is None:
        vec = _embed(text)

    meta = raw.get("meta")
    now = time.time()

    def _num(key: str, fallback: float) -> float:
        value = raw.get(key, fallback)
        return float(value) if isinstance(value, (int, float)) else fallback

    return {
        "id": raw.get("id") or f"doc_{uuid.uuid4().hex}",
        "text": text,
        "meta": meta if isinstance(meta, dict) else {},
        "vec": vec,
        "created_at": _num("created_at", now),
        "accessed_at": _num("accessed_at", now),
        "access_count": int(_num("access_count", 0)),
    }


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
        self._dirty = False
        self._try_chroma()
        self._load_persistent()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def _cache_path(self) -> Path:
        return self.persist_dir / f"{self.name}_cache.json"

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

    # ── persistence (offline backend only) ────────────────────────────────────
    def _load_persistent(self) -> None:
        """Load in-memory docs from the local JSON cache."""
        if self._backend != "offline":
            return
        raw = read_json(self._cache_path)
        if raw is None:
            return

        if isinstance(raw, dict):
            entries = raw.get("docs")
            version = raw.get("schema_version")
        elif isinstance(raw, list):
            # Pre-versioned layout: a bare list of docs.
            entries, version = raw, 1
        else:
            entries, version = None, None

        if not isinstance(entries, list):
            logger.warning("ignoring unrecognised vector cache at %s", self._cache_path)
            return

        docs: list[dict[str, Any]] = []
        dropped = 0
        for entry in entries:
            doc = _doc_from_json(entry)
            if doc is None:
                dropped += 1
            else:
                docs.append(doc)

        self._docs = docs
        if dropped:
            logger.warning("dropped %d malformed entr%s from %s",
                           dropped, "y" if dropped == 1 else "ies", self._cache_path)
        if version != _SCHEMA_VERSION:
            # Loaded fine, but rewrite in the current layout on the next flush.
            self._dirty = True
        logger.info("loaded %d cached memories from %s", len(docs), self._cache_path)

    def _flush_locked(self, force: bool = False) -> None:
        """Write the cache if anything changed.  Caller must hold ``_lock``."""
        if self._backend != "offline":
            return
        if not (self._dirty or force):
            return
        payload = {
            "schema_version": _SCHEMA_VERSION,
            # Deliberately not tagged with the dense embedder: this cache holds
            # the *sparse* vectors that offline search uses, which do not depend
            # on it.  Naming it here would also drag the transformer load into
            # a code path that never needs a dense vector.
            "docs": [_doc_to_json(d) for d in self._docs],
        }
        if atomic_write_json(self._cache_path, payload):
            self._dirty = False
        else:
            logger.warning("vector memory cache not persisted: %s", self._cache_path)

    def flush(self) -> None:
        """Persist pending changes now (e.g. on shutdown)."""
        with self._lock:
            self._flush_locked()

    # ── public API ────────────────────────────────────────────────────────────
    def add(self, text: str, metadata: dict[str, Any] | None = None,
            id: str | None = None, mem_type: str = "semantic",
            importance: float = 0.5) -> str:
        """Add a memory with type classification and importance score."""
        cid = id or f"doc_{uuid.uuid4().hex}"
        now = time.time()
        meta = {"mem_type": mem_type, "importance": importance,
                **(metadata or {})}

        if self._backend == "chromadb":
            # created_at must live in the metadata or query-time episodic decay
            # has nothing to work from.
            meta.setdefault("created_at", now)
            meta.setdefault("access_count", 0)
            meta.setdefault("embedder", embedder_name())
            self._collection.add(  # type: ignore[union-attr]
                documents=[text],
                metadatas=[meta],
                embeddings=[_get_embedding(text)],
                ids=[cid],
            )
            return cid

        doc = {
            "id": cid, "text": text, "meta": meta,
            "vec": _embed(text),  # sparse vector for offline search
            "created_at": now,
            "accessed_at": now,
            "access_count": 0,
        }
        with self._lock:
            self._docs.append(doc)
            self._dirty = True
            self._flush_locked()
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
            res = self._collection.query(  # type: ignore[union-attr]
                # Must mirror how add() stores vectors.  Passing query_texts=
                # instead would embed with chromadb's own default model, i.e.
                # search a different vector space than the one we wrote.
                query_embeddings=[_get_embedding(text)],
                n_results=max(1, top_k * 3),
                where=where,
            )
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]

            now = time.time()
            scored: list[tuple[float, dict[str, Any], str]] = []
            for cid, d, m, dist in zip(ids, docs, metas, dists):
                m = dict(m or {})
                # Vectors are L2-normalised, so squared-L2 distance maps back to
                # cosine as cos = 1 - dist/2.
                sim = max(0.0, 1.0 - float(dist) / 2.0)
                imp = float(m.get("importance", 0.5))
                if m.get("mem_type") == "episodic":
                    created = float(m.get("created_at", now))
                    age_days = (now - created) / 86400
                    imp *= math.pow(0.5, age_days / self._half_life_days)
                access_count = int(m.get("access_count", 0)) + 1
                imp += min(access_count / 10.0, 0.5)
                final_score = sim * (0.5 + imp)  # importance boosts relevance
                scored.append((final_score, {
                    "id": cid, "text": d, "metadata": m,
                    "score": float(sim), "final_score": float(final_score),
                    "access_count": access_count,
                }, cid))

            scored.sort(key=lambda x: x[0], reverse=True)
            hits = [s for s in scored[:top_k] if s[0] > 0.0]
            self._touch_chroma(hits, now)
            return [h[1] for h in hits]
        except Exception as exc:
            logger.warning("chromadb query failed, using offline: %s", exc)
            return self._query_offline(text, top_k, mem_type)

    def _touch_chroma(self, hits: list[tuple[float, dict[str, Any], str]],
                      now: float) -> None:
        """Record the access so frequently-recalled memories rank higher.

        Best-effort: a bookkeeping failure must never break recall.
        """
        if not hits:
            return
        try:
            self._collection.update(  # type: ignore[union-attr]
                ids=[h[2] for h in hits],
                metadatas=[
                    {**h[1]["metadata"],
                     "access_count": h[1]["access_count"],
                     "accessed_at": now}
                    for h in hits
                ],
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping only
            logger.debug("could not record access counts in chromadb: %s", exc)

    def _query_offline(self, text: str, top_k: int, mem_type: str | None) -> list[dict[str, Any]]:
        q = _embed(text)
        now = time.time()
        results: list[dict[str, Any]] = []

        with self._lock:
            candidates = [d for d in self._docs
                          if mem_type is None or d["meta"].get("mem_type") == mem_type]
            scored = [(d["vec"].cos(q), d) for d in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)

            for s, d in scored[:top_k]:
                if s <= 0.0:
                    continue
                meta = d["meta"]
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
                    "id": d["id"], "text": d["text"], "metadata": meta,
                    "score": float(s), "final_score": float(final_score),
                    "access_count": access_count,
                })

            # Access counts changed, but a *read* does not earn a whole-file
            # rewrite -- the next mutation (or flush()) will carry them.
            if results:
                self._dirty = True

        return results

    def count(self, mem_type: str | None = None) -> int:
        if self._backend == "chromadb":
            try:
                if mem_type:
                    result = self._collection.get(  # type: ignore[union-attr]
                        where={"mem_type": mem_type}, include=["metadatas"],
                    )
                    return len((result or {}).get("ids") or [])
                return int(self._collection.count())  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                logger.warning("chromadb count failed: %s", exc)
                return 0
        with self._lock:
            if mem_type:
                return sum(1 for d in self._docs if d["meta"].get("mem_type") == mem_type)
            return len(self._docs)

    def decay(self, half_life_days: float | None = None) -> int:
        """Apply memory decay to episodic memories and return count of decayed."""
        half_life = half_life_days or self._half_life_days
        if self._backend == "chromadb":
            return self._decay_chroma(half_life)

        now = time.time()
        decayed = 0
        with self._lock:
            keep: list[dict[str, Any]] = []
            for d in self._docs:
                meta = d["meta"]
                if meta.get("mem_type") != "episodic":
                    keep.append(d)
                    continue
                created = d.get("created_at", now)
                age_days = (now - created) / 86400
                decay_factor = math.pow(0.5, age_days / half_life)
                old_imp = float(meta.get("importance", 0.5))
                new_imp = old_imp * decay_factor
                if new_imp < 0.05 and old_imp >= 0.05:
                    # Memory is now stale — drop it by omission.  Removing from
                    # the list being iterated would skip the next element.
                    decayed += 1
                    continue
                meta["importance"] = new_imp
                keep.append(d)

            if decayed or keep != self._docs:
                self._docs = keep
                self._dirty = True
                self._flush_locked()
        return decayed

    def _decay_chroma(self, half_life: float) -> int:
        """Same decay pass over the chromadb backend."""
        try:
            result = self._collection.get(  # type: ignore[union-attr]
                where={"mem_type": "episodic"}, include=["metadatas"],
            ) or {}
            ids = result.get("ids") or []
            metas = result.get("metadatas") or []
            now = time.time()
            stale: list[str] = []
            updated_ids: list[str] = []
            updated_metas: list[dict[str, Any]] = []

            for cid, meta in zip(ids, metas):
                meta = dict(meta or {})
                created = float(meta.get("created_at", now))
                age_days = (now - created) / 86400
                old_imp = float(meta.get("importance", 0.5))
                new_imp = old_imp * math.pow(0.5, age_days / half_life)
                if new_imp < 0.05 and old_imp >= 0.05:
                    stale.append(cid)
                else:
                    meta["importance"] = new_imp
                    updated_ids.append(cid)
                    updated_metas.append(meta)

            if updated_ids:
                self._collection.update(ids=updated_ids, metadatas=updated_metas)  # type: ignore[union-attr]
            if stale:
                self._collection.delete(ids=stale)  # type: ignore[union-attr]
            return len(stale)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chromadb decay failed: %s", exc)
            return 0

    def forget(self, doc_id: str) -> bool:
        if self._backend == "chromadb":
            try:
                self._collection.delete(ids=[doc_id])  # type: ignore[union-attr]
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("chromadb delete failed for %s: %s", doc_id, exc)
                return False
        with self._lock:
            before = len(self._docs)
            self._docs = [d for d in self._docs if d["id"] != doc_id]
            if len(self._docs) != before:
                self._dirty = True
            self._flush_locked()
            return len(self._docs) != before

    def clear(self) -> None:
        if self._backend == "chromadb":
            try:
                self._client.delete_collection(self.name)  # type: ignore[union-attr]
                self._collection = self._client.get_or_create_collection(name=self.name)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                logger.warning("chromadb clear failed: %s", exc)
            return
        with self._lock:
            self._docs.clear()
            self._dirty = False
            self._cache_path.unlink(missing_ok=True)

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
