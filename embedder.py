"""
Lightweight embedding: TF-IDF + TruncatedSVD → 384-dim vectors.

Pure numpy/scipy — no ONNX, no PyTorch, no external models.
Fits a vocabulary and SVD transform on first batch of text, then
reuses them for all subsequent embeddings.

Usage:
    ef = NumpyEmbedder()
    vectors = ef.embed(["hello world", "another document"])
    # vectors shape: (2, 384)
"""

import hashlib
import json
import logging
import os
import pickle
import re
import string
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

# Embedding dimension — matches MiniLM-L6-v2 so it's a drop-in replacement
_EMBED_DIM = 384

# How many top terms to keep in vocabulary (prevents unbounded memory growth)
_MAX_VOCAB = 32_768

# Min / max document frequency for vocabulary pruning
_MIN_DF = 2
_MAX_DF = 0.85

# Regular expression for tokenization
_TOKEN_RE = re.compile(r"(?u)\b\w[\w'-]*\w\b|\b\w\b")


class NumpyEmbedder:
    """TF-IDF → SVD embedding function. Fits incrementally."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._vocab: dict[str, int] = {}  # term → index
        self._idf: Optional[np.ndarray] = None  # shape: (vocab_size,)
        self._svd_components: Optional[np.ndarray] = None  # shape: (embed_dim, vocab_size)
        self._svd_mean: Optional[np.ndarray] = None  # shape: (embed_dim,)
        self._fitted = False
        self._doc_count = 0
        self._df: dict[str, int] = {}  # term → document frequency

        if model_dir:
            self._load()

    # ── Public API ──────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns float32 array shape (n, 384)."""
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        if not self._fitted:
            # First call: fit on these texts
            self._fit(texts)

        # Transform texts → TF-IDF vectors → SVD reduced
        tfidf = self._transform(texts)
        if self._svd_components is None or self._svd_mean is None:
            return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
        # Project with SVD
        centered = tfidf - self._svd_mean
        return (centered @ self._svd_components.T).astype(np.float32)

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, directory: str) -> None:
        """Persist the model to a directory."""
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self._vocab,
            "idf": self._idf.tolist() if self._idf is not None else None,
            "svd_components": self._svd_components.tolist() if self._svd_components is not None else None,
            "svd_mean": self._svd_mean.tolist() if self._svd_mean is not None else None,
            "doc_count": self._doc_count,
            "df": self._df,
        }
        with open(path / "embedder.json", "w") as f:
            json.dump(data, f)
        logger.info("Embedder saved to %s (%d terms)", self.model_dir, len(self._vocab))

    def _load(self) -> None:
        """Load model from directory if exists."""
        path = Path(self.model_dir) / "embedder.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self._vocab = data.get("vocab", {})
        self._idf = np.array(data["idf"]) if data.get("idf") else None
        self._svd_components = np.array(data["svd_components"]) if data.get("svd_components") else None
        self._svd_mean = np.array(data["svd_mean"]) if data.get("svd_mean") else None
        self._doc_count = data.get("doc_count", 0)
        self._df = data.get("df", {})
        self._fitted = bool(self._vocab)
        if self._fitted:
            logger.info(
                "Embedder loaded from %s (%d terms, %d docs seen)",
                self.model_dir, len(self._vocab), self._doc_count
            )

    def _fit(self, texts: list[str]) -> None:
        """Build vocabulary and SVD transform from a batch of texts."""
        # Tokenize and build vocab + DF
        tokenized = []
        for t in texts:
            tokens = self._tokenize(t)
            tokenized.append(tokens)
            unique = set(tokens)
            for tok in unique:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._doc_count += len(texts)

        # Prune vocabulary: filter by DF range
        n_docs = self._doc_count
        filtered = {}
        for term, df in self._df.items():
            freq = df / n_docs
            if df >= _MIN_DF and freq <= _MAX_DF:
                filtered[term] = len(filtered)
        # Keep only top _MAX_VOCAB terms
        if len(filtered) > _MAX_VOCAB:
            # Sort by DF descending, keep top _MAX_VOCAB
            sorted_terms = sorted(filtered.items(), key=lambda x: -self._df.get(x[0], 0))
            filtered = {t: i for i, (t, _) in enumerate(sorted_terms[:_MAX_VOCAB])}

        self._vocab = filtered
        vocab_size = len(self._vocab)

        if vocab_size == 0:
            logger.warning("Empty vocabulary — no training data")
            self._fitted = True
            return

        # Compute IDF
        idf = np.zeros(vocab_size, dtype=np.float64)
        for term, idx in self._vocab.items():
            df = self._df.get(term, 0)
            idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1.0
        self._idf = idf

        # Build TF-IDF matrix for the input texts
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(tokenized):
            tf: dict[str, int] = {}
            for tok in tokens:
                if tok in self._vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            n_tokens = len(tokens) if tokens else 1
            for term, count in tf.items():
                rows.append(i)
                cols.append(self._vocab[term])
                vals.append((count / n_tokens) * idf[self._vocab[term]])
        tfidf = csr_array((vals, (rows, cols)), shape=(len(texts), vocab_size), dtype=np.float64)

        # Run SVD: reduce to embed_dim
        k = min(_EMBED_DIM, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if k < 1:
            logger.warning("Not enough data for SVD (k=%d)", k)
            self._fitted = True
            return

        u, s, vt = svds(tfidf, k=k)
        # Sort by singular value descending
        idx = np.argsort(-s)
        s = s[idx]
        vt = vt[idx]
        # Build components matrix: (embed_dim, vocab_size)
        components = np.zeros((_EMBED_DIM, vocab_size), dtype=np.float64)
        k_use = min(k, _EMBED_DIM)
        components[:k_use] = vt[:k_use]
        self._svd_components = components

        # Compute mean for centering
        self._svd_mean = tfidf.mean(axis=0)
        if hasattr(self._svd_mean, "A1"):
            self._svd_mean = np.asarray(self._svd_mean).flatten()
        else:
            self._svd_mean = self._svd_mean.flatten()

        self._fitted = True
        logger.info(
            "Embedder fitted: %d terms, %d docs, SVD k=%d → %d dims",
            vocab_size, n_docs, k_use, _EMBED_DIM
        )

    def _transform(self, texts: list[str]) -> csr_array:
        """Transform texts to TF-IDF vectors using fitted vocab/IDF."""
        if not self._vocab or self._idf is None:
            return csr_array((len(texts), 1), dtype=np.float64)

        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            tokens = self._tokenize(t)
            tf: dict[str, int] = {}
            for tok in tokens:
                if tok in self._vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            n_tokens = len(tokens) if tokens else 1
            for term, count in tf.items():
                idx = self._vocab[term]
                rows.append(i)
                cols.append(idx)
                vals.append((count / n_tokens) * self._idf[idx])
        return csr_array(
            (vals, (rows, cols)),
            shape=(len(texts), len(self._vocab)),
            dtype=np.float64,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words."""
        text = text.lower()
        # Remove punctuation (keep apostrophes and hyphens within words)
        text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "").replace("-", "")))
        tokens = _TOKEN_RE.findall(text)
        return [t for t in tokens if len(t) >= 2 or t.isalpha()]


# Convenience singleton
_global_embedder: Optional[NumpyEmbedder] = None


def get_embedder(model_dir: Optional[str] = None) -> NumpyEmbedder:
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = NumpyEmbedder(model_dir=model_dir)
    return _global_embedder
