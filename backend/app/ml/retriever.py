"""
Hybrid retriever: dense (embedding) + lexical (TF-IDF BM25-like) + reranking.
Supports document diversity to avoid one paper dominating results.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.embedder import batch_cosine_sim, encode, json_to_embedding

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    paper_id: str
    text: str
    page: int
    section: str
    dense_score: float
    lexical_score: float
    rerank_score: float
    rank: int


def _tfidf_scores(query: str, texts: list[str]) -> np.ndarray:
    """TF-IDF lexical similarity scores."""
    if not texts:
        return np.array([])
    vectorizer = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
    try:
        corpus = [query] + texts
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = tfidf_matrix[0]
        doc_vecs = tfidf_matrix[1:]
        scores = cosine_similarity(query_vec, doc_vecs).flatten()
        return scores
    except Exception:
        return np.zeros(len(texts))


def _rerank_score(query_emb: np.ndarray, chunk_embs: list[np.ndarray]) -> np.ndarray:
    """Simple reranking using dot product of normalized embeddings."""
    if not chunk_embs:
        return np.array([])
    matrix = np.stack(chunk_embs)
    return batch_cosine_sim(query_emb, matrix)


def retrieve(
    query: str,
    chunks: list[dict],  # list of {chunk_id, paper_id, text, page, section, embedding_json}
    top_k: int = 12,
    dense_weight: float = 0.6,
    lexical_weight: float = 0.4,
    max_per_paper: int = 3,
) -> list[RetrievalResult]:
    """
    Hybrid retrieval with document diversity.
    chunks: list of dicts with keys: chunk_id, paper_id, text, page, section, embedding_json
    Returns top_k results with at most max_per_paper from any single paper.
    """
    if not chunks:
        return []

    query_emb = encode([query])[0]
    texts = [c['text'] for c in chunks]

    # Dense scores
    chunk_embs = []
    for c in chunks:
        emb_json = c.get('embedding_json', '')
        if emb_json:
            chunk_embs.append(json_to_embedding(emb_json))
        else:
            chunk_embs.append(encode([c['text']])[0])

    dense_scores = batch_cosine_sim(query_emb, np.stack(chunk_embs))

    # Lexical scores
    lex_scores = _tfidf_scores(query, texts)
    if len(lex_scores) == 0:
        lex_scores = np.zeros(len(chunks))

    # Combined
    combined = dense_weight * dense_scores + lexical_weight * lex_scores

    # Sort descending
    order = np.argsort(combined)[::-1]

    # Apply document diversity
    results = []
    paper_counts: dict[str, int] = {}
    for idx in order:
        chunk = chunks[idx]
        pid = chunk['paper_id']
        if paper_counts.get(pid, 0) >= max_per_paper:
            continue
        paper_counts[pid] = paper_counts.get(pid, 0) + 1
        results.append(RetrievalResult(
            chunk_id=chunk['chunk_id'],
            paper_id=pid,
            text=chunk['text'],
            page=chunk.get('page', 1),
            section=chunk.get('section', ''),
            dense_score=float(dense_scores[idx]),
            lexical_score=float(lex_scores[idx]),
            rerank_score=float(combined[idx]),
            rank=len(results) + 1,
        ))
        if len(results) >= top_k:
            break

    return results
