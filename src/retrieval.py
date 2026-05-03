"""
Retrieval / grounding layer.

Embedder is kept as an interface so the agent and tool layer never know or
care whether similarity comes from TF-IDF, a local sentence-transformer, or a
hosted embeddings API -- you can swap TfidfEmbedder for a real embedding
model without touching agent.py or tools.py. TF-IDF is the default here
because it has zero network/model-download dependency, so this repo runs
anywhere out of the box; swapping in `sentence-transformers`
(all-MiniLM-L6-v2) is a one-file change documented in the README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Embedder(Protocol):
    """Interface any retrieval backend must satisfy."""

    def fit(self, documents: list[str]) -> None: ...
    def similarity(self, query: str, documents: list[str]) -> list[float]: ...


class TfidfEmbedder:
    """Default, dependency-light retrieval backend."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._doc_matrix = None

    def fit(self, documents: list[str]) -> None:
        self._doc_matrix = self._vectorizer.fit_transform(documents)

    def similarity(self, query: str, documents: list[str]) -> list[float]:
        if self._doc_matrix is None:
            raise RuntimeError("Embedder.fit() must be called before similarity()")
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._doc_matrix)[0]
        return scores.tolist()


class KnowledgeBase:
    """Loads the EV owner/service knowledge base and answers top-k retrieval queries."""

    def __init__(self, kb_path: str | Path, embedder: Embedder | None = None) -> None:
        self.kb_path = Path(kb_path)
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        raw = json.loads(self.kb_path.read_text())
        self.chunks = [Chunk(**item) for item in raw]
        corpus = [f"{c.title} {c.section} {c.text}" for c in self.chunks]
        self.embedder.fit(corpus)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[RetrievedChunk]:
        corpus = [f"{c.title} {c.section} {c.text}" for c in self.chunks]
        scores = self.embedder.similarity(query, corpus)
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        results = [RetrievedChunk(chunk=c, score=s) for c, s in ranked if s >= min_score]
        return results[:top_k]
