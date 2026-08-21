"""
JARVIS Retrieval-Augmented Generation (RAG) Pipeline — JARVIS 7.0.

Combines web search + vector memory + a reranker step before presenting
results to the model. This cuts down on noisy answers by filtering
irrelevant context before it reaches the LLM context window.

Pipeline:
  1. **Web search** — query the search plugin (or DuckDuckGo / Bing if no API).
  2. **Memory search** — semantic recall from vector memory for related facts.
  3. **Reranker** — an LLM call scores each candidate snippet for relevance
     to the original query (0-10). Only snippets scoring >= 6 are kept.
  4. **Synthesise** — pass the top-N reranked results to the LLM with a
     "summarise these sources" prompt.

Example::

    from core.rag import rag

    answer = rag.answer("What are the health benefits of intermittent fasting?")
    # -> {"answer": "...", "sources": [...], "confidence": 0.85}
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from core.vector_memory import vector_db

logger = logging.getLogger("jarvis.rag")

DEFAULT_MIN_RELEVANCE = 6.0  # out of 10
DEFAULT_TOP_K = 5


@dataclass
class RAGSource:
    text: str
    metadata: dict[str, Any]
    relevance_score: float  # 0-10 from reranker
    source_type: str  # "web" or "memory"


class RAGPipeline:
    """Full RAG pipeline: search → recall → rerank → synthesise."""

    def __init__(self) -> None:
        self._reranker_cache: dict[str, float] = {}

    def answer(self, query: str, min_relevance: float = DEFAULT_MIN_RELEVANCE,
               top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Run the full RAG pipeline and return a synthesised answer."""
        start = time.time()

        # 1. Web search
        web_results = self._web_search(query, top_k=10)
        logger.debug("RAG: got %d web results", len(web_results))

        # 2. Memory recall
        mem_results = self._memory_search(query, top_k=top_k)
        logger.debug("RAG: got %d memory results", len(mem_results))

        # 3. Rerank all results
        all_sources = web_results + mem_results
        reranked = self._rerank(query, all_sources)
        filtered = [s for s in reranked if s.relevance_score >= min_relevance][:top_k]

        if not filtered:
            return {
                "answer": "I couldn't find reliable information on that topic. Would you like me to search differently?",
                "sources": [],
                "confidence": 0.0,
                "elapsed_s": round(time.time() - start, 2),
                "query": query,
            }

        # 4. Synthesise answer
        answer = self._synthesise(query, filtered)
        confidence = min(sum(s.relevance_score for s in filtered) / (len(filtered) * 10), 1.0)

        return {
            "answer": answer,
            "sources": [
                {
                    "text": s.text[:200],
                    "type": s.source_type,
                    "relevance": round(s.relevance_score, 1),
                    "metadata": s.metadata,
                }
                for s in filtered
            ],
            "confidence": round(confidence, 2),
            "elapsed_s": round(time.time() - start, 2),
            "query": query,
        }

    def _web_search(self, query: str, top_k: int = 10) -> list[RAGSource]:
        """Query web search and convert results to RAGSource objects."""
        try:
            from actions.web_search import _gemini_search
            results = _gemini_search(query, detailed=True)
            sources: list[RAGSource] = []
            for r in results.get("results", [])[:top_k]:
                sources.append(RAGSource(
                    text=r.get("content", r.get("title", "")),
                    metadata={"title": r.get("title", ""), "url": r.get("url", "")},
                    relevance_score=5.0,  # default before reranking
                    source_type="web",
                ))
            return sources
        except Exception as exc:
            logger.debug("RAG web search failed: %s", exc)
            return []

    def _memory_search(self, query: str, top_k: int = 5) -> list[RAGSource]:
        """Search vector memory for relevant stored facts."""
        try:
            results = vector_db.query(query, top_k=top_k)
            return [
                RAGSource(
                    text=r["text"],
                    metadata=r.get("metadata", {}),
                    relevance_score=5.0,
                    source_type="memory",
                )
                for r in results
            ]
        except Exception as exc:
            logger.debug("RAG memory search failed: %s", exc)
            return []

    def _rerank(self, query: str, sources: list[RAGSource]) -> list[RAGSource]:
        """Use an LLM to rerank sources by relevance to the query."""
        if not sources:
            return []

        # Build a batch for efficient reranking
        snippets = "\n".join(
            f"[{i}] {s.text[:300]}"
            for i, s in enumerate(sources)
        )

        system = (
            "You are a relevance-ranking assistant. Given a user query and a "
            "list of text snippets, score each snippet 0-10 for how relevant it "
            "is to answering the query. Return ONLY a JSON array of integers "
            "(one score per snippet, in order)."
        )
        prompt = (
            f"QUERY: {query}\n\n"
            f"SNIPPETS:\n{snippets}\n\n"
            f"Rate each snippet 0-10 for relevance to the query."
        )

        try:
            from core.llm_client import call_llm_text
            raw = call_llm_text(prompt, system=system, timeout=30)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                scores = json.loads(raw[start:end + 1])
            else:
                scores = json.loads(raw)
        except Exception:
            # Fallback: uniform score
            scores = [5.0] * len(sources)

        if isinstance(scores, list) and len(scores) == len(sources):
            for i, score in enumerate(scores):
                try:
                    sources[i].relevance_score = float(score)
                except (ValueError, TypeError):
                    pass

        return sorted(sources, key=lambda s: s.relevance_score, reverse=True)

    def _synthesise(self, query: str, sources: list[RAGSource]) -> str:
        """Generate a coherent answer from the top reranked sources."""
        source_texts = "\n\n".join(
            f"Source {i+1}: {s.text}"
            for i, s in enumerate(sources)
        )

        system = (
            "You are a helpful research assistant. Synthesize a clear, concise "
            "answer from the user's sources. If the sources conflict, mention "
            "both sides. Cite sources with [1], [2], etc. If uncertain, say so."
        )
        prompt = (
            f"QUERY: {query}\n\n"
            f"SOURCES:\n{source_texts}\n\n"
            f"Provide a well-structured answer based on these sources."
        )

        try:
            from core.llm_client import call_llm_text
            return call_llm_text(prompt, system=system, timeout=30).strip()
        except Exception:
            try:
                from core.gemini_text import generate
                return generate(prompt, system=system, timeout=30).strip()
            except Exception:
                return "Unable to synthesise an answer at this time."


# Process-wide instance.
rag = RAGPipeline()
