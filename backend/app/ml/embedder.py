"""
Embedding model wrapper.
Uses sentence-transformers when available; falls back to TF-IDF cosine similarity.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_sentence_transformer = None
_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None
_tfidf_docs: list[str] = []


def _get_st():
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            from app.config import settings
            _sentence_transformer = SentenceTransformer(settings.embedding_model)
            logger.info('Loaded SentenceTransformer: %s', settings.embedding_model)
        except Exception as e:
            logger.warning('SentenceTransformer unavailable: %s. Using TF-IDF fallback.', e)
            _sentence_transformer = False
    return _sentence_transformer if _sentence_transformer is not False else None


def encode(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into embeddings. Shape: (N, D)."""
    if not texts:
        return np.zeros((0, 384))
    st = _get_st()
    if st:
        return st.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    # TF-IDF fallback
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        _tfidf_vectorizer = TfidfVectorizer(max_features=4096)
        vecs = _tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)
    else:
        vecs = _tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs / norms


def embed_single(text: str) -> np.ndarray:
    return encode([text])[0]


def embed_to_json(text: str) -> str:
    return json.dumps(encode([text])[0].tolist())


def json_to_embedding(j: str) -> np.ndarray:
    if not j:
        return np.zeros(384)
    return np.array(json.loads(j), dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


def batch_cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of query vs each row of matrix."""
    q = query.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normed = matrix / norms
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    return (normed @ q_norm.T).flatten()
