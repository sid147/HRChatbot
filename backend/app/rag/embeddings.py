from __future__ import annotations

import math
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional import handling
    SentenceTransformer = None  # type: ignore

_model_lock = threading.Lock()
_cached_model: Optional[object] = None

# Fallback TF-IDF state
_tfidf_vocab_index: Dict[str, int] = {}
_tfidf_idf: Optional[np.ndarray] = None


def is_model_available() -> bool:
    return SentenceTransformer is not None


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer: lowercase and split on non-alphanumeric
    import re

    text = re.sub(r"\s+", " ", text.strip().lower())
    tokens = re.split(r"[^a-z0-9\+\.#]+", text)
    return [t for t in tokens if t]


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not is_model_available():
        raise RuntimeError(
            "sentence-transformers is not installed. Install it to enable semantic search."
        )
    with _model_lock:
        if _cached_model is None:
            _cached_model = SentenceTransformer(model_name)
    return _cached_model


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return (matrix / norms).astype(np.float32)


def _build_tfidf(texts: List[str]) -> Tuple[Dict[str, int], np.ndarray]:
    # Build vocabulary and IDF
    doc_freq: Dict[str, int] = {}
    for text in texts:
        unique_tokens = set(_tokenize(text))
        for tok in unique_tokens:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    vocab = {tok: idx for idx, tok in enumerate(sorted(doc_freq.keys()))}
    n_docs = max(1, len(texts))
    idf = np.zeros((len(vocab),), dtype=np.float32)
    for tok, df in doc_freq.items():
        idx = vocab[tok]
        idf[idx] = math.log((n_docs + 1) / (df + 1)) + 1.0
    return vocab, idf


def _tfidf_vectorize(texts: List[str]) -> np.ndarray:
    assert _tfidf_idf is not None and _tfidf_vocab_index, "TF-IDF is not initialized"
    vocab = _tfidf_vocab_index
    idf = _tfidf_idf
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = _tokenize(text)
        if not tokens:
            continue
        tf: Dict[int, int] = {}
        for tok in tokens:
            idx = vocab.get(tok)
            if idx is None:
                continue
            tf[idx] = tf.get(idx, 0) + 1
        if not tf:
            continue
        max_tf = max(tf.values())
        for idx, count in tf.items():
            tf_weight = 0.5 + 0.5 * (count / max_tf)
            matrix[i, idx] = tf_weight * idf[idx]
    return _l2_normalize(matrix)


def initialize_fallback_tfidf(corpus_texts: List[str]) -> None:
    global _tfidf_vocab_index, _tfidf_idf
    _tfidf_vocab_index, _tfidf_idf = _build_tfidf(corpus_texts)


def compute_embeddings(texts: List[str]) -> np.ndarray:
    if is_model_available():
        model = load_embedding_model()
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    # Fallback TF-IDF: must be initialized by calling initialize_fallback_tfidf with corpus first
    if _tfidf_idf is None or not _tfidf_vocab_index:
        raise RuntimeError(
            "TF-IDF fallback not initialized. Build employee embeddings first before embedding queries."
        )
    return _tfidf_vectorize(texts)


def embed_text(text: str) -> np.ndarray:
    return compute_embeddings([text])[0] 