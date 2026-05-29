"""Lightweight embedding: TF-IDF + TruncatedSVD -> 384-dim vectors."""

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

_EMBED_DIM = 384
_MAX_VOCAB = 32_768
_MIN_DF = 2
_MAX_DF = 0.85
_TOKEN_RE = re.compile(r"(?u)\b\w[\w'-]*\w\b|\b\w\b")


class NumpyEmbedder:
    """TF-IDF -> SVD embedding function. Fits incrementally."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._svd_components: Optional[np.ndarray] = None
        self._svd_mean: Optional[np.ndarray] = None
        self._fitted = False
        self._doc_count = 0
        self._df: dict[str, int] = {}

        if model_dir:
            self._load()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns float32 array shape (n, 384)."""
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        if not self._fitted:
            self._fit(texts)
        tfidf = self._transform(texts)
        if self._svd_components is None or self._svd_mean is None:
            return np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
        centered = tfidf - self._svd_mean
        return (centered @ self._svd_components.T).astype(np.float32)

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def save(self, directory: str) -> None:
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
            logger.info("Embedder loaded from %s (%d terms, %d docs seen)", self.model_dir, len(self._vocab), self._doc_count)

    def _fit(self, texts: list[str]) -> None:
        tokenized = []
        for t in texts:
            tokens = self._tokenize(t)
            tokenized.append(tokens)
            unique = set(tokens)
            for tok in unique:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._doc_count += len(texts)

        n_docs = self._doc_count
        filtered = {}
        for term, df in self._df.items():
            freq = df / n_docs
            if df >= _MIN_DF and freq <= _MAX_DF:
                filtered[term] = len(filtered)
        if len(filtered) > _MAX_VOCAB:
            sorted_terms = sorted(filtered.items(), key=lambda x: -self._df.get(x[0], 0))
            filtered = {t: i for i, (t, _) in enumerate(sorted_terms[:_MAX_VOCAB])}

        self._vocab = filtered
        vocab_size = len(self._vocab)

        if vocab_size == 0:
            logger.warning("Empty vocabulary")
            self._fitted = True
            return

        idf = np.zeros(vocab_size, dtype=np.float64)
        for term, idx in self._vocab.items():
            df = self._df.get(term, 0)
            idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1.0
        self._idf = idf

        rows, cols, vals = [], [], []
        for i, tokens in enumerate(tokenized):
            tf = {}
            for tok in tokens:
                if tok in self._vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            n_tokens = len(tokens) if tokens else 1
            for term, count in tf.items():
                rows.append(i)
                cols.append(self._vocab[term])
                vals.append((count / n_tokens) * idf[self._vocab[term]])
        tfidf = csr_array((vals, (rows, cols)), shape=(len(texts), vocab_size), dtype=np.float64)

        k = min(_EMBED_DIM, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if k < 1:
            logger.warning("Not enough data for SVD (k=%d)", k)
            self._fitted = True
            return

        u, s, vt = svds(tfidf, k=k)
        idx = np.argsort(-s)
        s = s[idx]
        vt = vt[idx]
        components = np.zeros((_EMBED_DIM, vocab_size), dtype=np.float64)
        k_use = min(k, _EMBED_DIM)
        components[:k_use] = vt[:k_use]
        self._svd_components = components

        self._svd_mean = tfidf.mean(axis=0)
        if hasattr(self._svd_mean, "A1"):
            self._svd_mean = np.asarray(self._svd_mean).flatten()
        else:
            self._svd_mean = self._svd_mean.flatten()

        self._fitted = True
        logger.info("Embedder fitted: %d terms, %d docs, SVD k=%d -> %d dims", vocab_size, n_docs, k_use, _EMBED_DIM)

    def _transform(self, texts: list[str]) -> csr_array:
        if not self._vocab or self._idf is None:
            return csr_array((len(texts), 1), dtype=np.float64)
        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            tokens = self._tokenize(t)
            tf = {}
            for tok in tokens:
                if tok in self._vocab:
                    tf[tok] = tf.get(tok, 0) + 1
            n_tokens = len(tokens) if tokens else 1
            for term, count in tf.items():
                idx = self._vocab[term]
                rows.append(i)
                cols.append(idx)
                vals.append((count / n_tokens) * self._idf[idx])
        return csr_array((vals, (rows, cols)), shape=(len(texts), len(self._vocab)), dtype=np.float64)

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation.replace("'", "").replace("-", "")))
        tokens = _TOKEN_RE.findall(text)
        return [t for t in tokens if len(t) >= 2 or t.isalpha()]


_global_embedder: Optional[NumpyEmbedder] = None
_onnx_warned: set = set()
_embedder_cache: dict = {}


def get_embedder(
    model_dir: Optional[str] = None,
    model: str = "numpy",
):
    """Return a cached embedder for the requested model.

    ``model="numpy"`` returns :class:`NumpyEmbedder` (default, always
    available). ``model="minilm"`` returns :class:`OnnxEmbedder`.
    ``model="embeddinggemma"`` returns :class:`EmbeddinggemmaONNX`.

    ONNX-based embedders require ``onnxruntime``, ``transformers``, and
    ``huggingface_hub`` at runtime. Missing dependencies raise
    ``ImportError`` when ``embed()`` is called (not at construction).
    """
    if model == "numpy":
        global _global_embedder
        if _global_embedder is None:
            _global_embedder = NumpyEmbedder(model_dir=model_dir)
        return _global_embedder

    key = f"{model}:{model_dir or ''}"
    cached = _embedder_cache.get(key)
    if cached is not None:
        return cached

    if model == "sentence":
        ef = SentenceTransformerEmbedder()
    elif model == "spacy":
        ef = SpacyGloveEmbedder()
    elif model == "minilm":
        ef = OnnxEmbedder()
    elif model == "embeddinggemma":
        ef = EmbeddinggemmaONNX()
    else:
        raise ValueError(f"Unknown embedder model: {model!r}")

    _embedder_cache[key] = ef
    return ef


# ── Sentence-Transformers embedder (via PyTorch, no ONNX) ───────────────


class SentenceTransformerEmbedder:
    """384-dim sentence transformer embedding via PyTorch (no onnxruntime).

    Uses ``sentence-transformers`` + PyTorch under the hood. First call to
    ``embed()`` downloads the model (cached by ``huggingface_hub``).

    Usage::

        ef = SentenceTransformerEmbedder()
        vectors = ef.embed(["hello world"])  # -> np.ndarray (n, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        self._lazy_load()
        return self._model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self._model_name)
        logger.info(
            "SentenceTransformerEmbedder loaded: model=%s dim=%d",
            self._model_name, _EMBED_DIM,
        )


# ── spaCy + GloVe embedder (no ONNX, no PyTorch) ────────────────────────


class SpacyGloveEmbedder:
    """300-dim GloVe embedding via spaCy, zero-padded to 384.

    Uses ``spacy`` with ``en_core_web_md`` (or ``en_core_web_lg``) word
    vectors. Document vectors are averaged word vectors, L2-normalized.
    Output is zero-padded from 300→384 to match FAISS index dimension.

    First call to ``embed()`` loads the model (downloaded on ``spacy
    download``).

    Usage::

        ef = SpacyGloveEmbedder()
        vectors = ef.embed(["hello world"])  # -> np.ndarray (n, 384)
    """

    _GLOVE_DIM = 300

    def __init__(self, model_name: str = "en_core_web_md"):
        self._model_name = model_name
        self._nlp = None

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def is_fitted(self) -> bool:
        return self._nlp is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        self._lazy_load()
        docs = list(self._nlp.pipe(texts))
        vectors = np.zeros((len(texts), _EMBED_DIM), dtype=np.float32)
        for i, doc in enumerate(docs):
            if doc.vector_norm:
                glove = doc.vector / doc.vector_norm
                vectors[i, :self._GLOVE_DIM] = glove[:self._GLOVE_DIM]
        return vectors

    def _lazy_load(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "SpacyGloveEmbedder requires spacy. "
                "Install with: pip install spacy && python -m spacy download en_core_web_md"
            ) from e
        self._nlp = spacy.load(self._model_name)
        logger.info(
            "SpacyGloveEmbedder loaded: model=%s dim=%d (zero-padded to %d)",
            self._model_name, self._GLOVE_DIM, _EMBED_DIM,
        )


# ── Hardware acceleration support ────────────────────────────────────────


_ACCELERATOR_MAP: dict[str, tuple[str, str]] = {
    "cuda": ("CUDAExecutionProvider", "mempalace[gpu]"),
    "coreml": ("CoreMLExecutionProvider", "mempalace[coreml]"),
    "dml": ("DmlExecutionProvider", "mempalace[dml]"),
}
_AUTO_ORDER = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider"]


def _available_onnx_providers() -> set:
    """Return the set of ONNX Runtime providers available in the installed package."""
    try:
        import onnxruntime as ort

        return set(ort.get_available_providers())
    except ImportError:
        return set()


def resolve_embedding_device(device: Optional[str] = None) -> tuple[list[str], str]:
    """Resolve ONNX Runtime provider list + effective device label.

    ``device=None`` returns CPU fallback. ``"auto"`` probes available
    providers and picks the fastest available (CUDA > CoreML > DirectML > CPU).

    Returns ``(provider_list, device_label)`` where ``device_label`` is
    ``"cpu"``, ``"cuda"``, ``"coreml"``, or ``"dml"``.
    """
    device = (device or "cpu").strip().lower()

    available = _available_onnx_providers()
    if not available:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for name in _AUTO_ORDER:
            if name in available:
                return ([name, "CPUExecutionProvider"], name.lower().replace("executionprovider", ""))
        return (["CPUExecutionProvider"], "cpu")

    if device == "cpu":
        return (["CPUExecutionProvider"], "cpu")

    entry = _ACCELERATOR_MAP.get(device)
    if entry is None:
        if device not in _onnx_warned:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _onnx_warned.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred, extra = entry
    if preferred not in available:
        if device not in _onnx_warned:
            logger.warning(
                "embedding_device=%r requested but %s not available — "
                "falling back to CPU. Install onnxruntime with %s.",
                device, preferred, extra,
            )
            _onnx_warned.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return ([preferred, "CPUExecutionProvider"], device)


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved embedding device.

    Tries ONNX Runtime first (GPU acceleration when available), then
    reports ``"numpy_tfidf_svd"``, ``"sentence_transformers"``, or
    ``"spacy_glove"`` depending on which backend is used.
    """
    providers, effective = resolve_embedding_device(device)
    if effective != "cpu":
        return f"onnx_{effective}"
    try:
        import onnxruntime as ort

        if ort.get_available_providers():
            return "onnx_cpu"
    except ImportError:
        pass
    return "numpy_tfidf_svd"


# ── ONNX embedder (optional — requires onnxruntime + huggingface_hub) ────


class OnnxEmbedder:
    """ONNX-based embedding function using all-MiniLM-L6-v2.

    384-dim output, compatible with the FAISS index dimension. Requires
    ``onnxruntime`` and ``huggingface_hub`` at runtime (imports are lazy).

    Usage::

        ef = OnnxEmbedder(device="auto")   # auto-select GPU
        vectors = ef.embed(["hello world"])  # -> np.ndarray (n, 384)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self._providers, self._device_label = resolve_embedding_device(device)
        self._model_name = model_name
        self._session = None
        self._tokenizer = None

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def device_label(self) -> str:
        return self._device_label

    def embed(self, texts: list[str]) -> "np.ndarray":
        """Embed texts. Returns float32 array shape (n, 384)."""
        if not texts:
            return __import__("numpy").zeros((0, _EMBED_DIM), dtype=np.float32)
        self._lazy_load()
        np = __import__("numpy")
        encs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np",
        )
        outputs = self._session.run(
            None, {
                "input_ids": encs["input_ids"],
                "attention_mask": encs["attention_mask"],
            }
        )
        # last_hidden_state mean pooling
        token_emb = outputs[0]
        mask = encs["attention_mask"][:, :, np.newaxis].astype(np.float32)
        sum_emb = np.sum(token_emb * mask, axis=1)
        sum_mask = np.sum(mask, axis=1)
        pooled = sum_emb / np.clip(sum_mask, a_min=1e-9, a_max=None)
        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
        return (pooled / norms).astype(np.float32)

    def _lazy_load(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "OnnxEmbedder requires onnxruntime, transformers, and numpy. "
                "Install with: pip install onnxruntime transformers numpy"
            ) from e

        model_id = f"sentence-transformers/{self._model_name}"
        # ONNX export path: download the ONNX model or use the transformer model
        # with optimum. For simplicity, load the full model via optimum.
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction

            self._session = ORTModelForFeatureExtraction.from_pretrained(
                model_id, export=True, provider=self._providers[0]
            )
            # ORTModelForFeatureExtraction wraps the session; extract it
            self._session = self._session.session
        except ImportError:
            # Fallback: try direct ONNX model on HF hub
            self._session = self._load_direct_onnx(model_id, ort, np)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info(
            "OnnxEmbedder loaded: model=%s device=%s",
            self._model_name, self._device_label,
        )

    def _load_direct_onnx(
        self, model_id: str, ort: "Any", np: "Any"
    ) -> "Any":
        """Attempt to load a pre-exported ONNX model from HuggingFace hub.

        Falls back to onnx subfolder if available.
        """
        from huggingface_hub import hf_hub_download

        try:
            model_path = hf_hub_download(
                model_id, subfolder="onnx", filename="model.onnx"
            )
        except Exception:
            model_path = hf_hub_download(
                model_id.replace("sentence-transformers/", "")
                .replace("-", "-") + "-ONNX",
                filename="model.onnx",
            )
        return ort.InferenceSession(model_path, providers=self._providers)


# ── Embeddinggemma-300m ONNX (optional — requires onnxruntime + huggingface_hub) ────


_EMBEDDINGGEMMA_REPO = "onnx-community/embeddinggemma-300m-ONNX"
_EMBEDDINGGEMMA_ONNX = "model_quantized.onnx"
_EMBEDDINGGEMMA_PREFIX = "task: sentence similarity | query: "
_EMBEDDINGGEMMA_DIM = 384
_EMBEDDINGGEMMA_MAX_LEN = 2048


class EmbeddinggemmaONNX:
    """Multilingual ONNX embedder using embeddinggemma-300m (q8, MRL -> 384d).

    100+ language support with cross-lingual cosine ~0.88 vs 0.35 for
    all-MiniLM-L6-v2. The ~300 MB model downloads from HuggingFace on
    first use and is cached by ``huggingface_hub``.

    Requires ``onnxruntime``, ``huggingface_hub``, and ``tokenizers``
    at runtime. All imports are lazy — missing dependencies raise
    ``ImportError`` from ``embed()``, not from construction.

    Compatible with kai's 384-dim FAISS index via Matryoshka truncation.
    """

    def __init__(self, preferred_providers=None):
        self._providers = (
            list(preferred_providers) if preferred_providers else ["CPUExecutionProvider"]
        )
        self._session = None
        self._tokenizer = None
        self._np = None
        self._output_idx = None

    @property
    def dimension(self) -> int:
        return _EMBEDDINGGEMMA_DIM

    def embed(self, texts: list[str]) -> "np.ndarray":
        """Embed texts. Returns float32 array shape (n, 384)."""
        if not texts:
            return __import__("numpy").zeros((0, _EMBEDDINGGEMMA_DIM), dtype=np.float32)
        self._lazy_load()
        np = self._np
        prefixed = [_EMBEDDINGGEMMA_PREFIX + t for t in texts]
        encs = self._tokenizer.encode_batch(prefixed)
        input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        sent_emb = outputs[self._output_idx][:, :_EMBEDDINGGEMMA_DIM]
        norms = np.linalg.norm(sent_emb, axis=1, keepdims=True) + 1e-12
        return (sent_emb / norms).astype(np.float32)

    def _lazy_load(self) -> None:
        if self._session is not None:
            return
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "EmbeddinggemmaONNX requires onnxruntime, huggingface_hub, "
                "and tokenizers. Install with: pip install onnxruntime "
                "huggingface_hub tokenizers"
            ) from e

        logger.info(
            "Downloading %s/%s (cached after first run)...",
            _EMBEDDINGGEMMA_REPO, _EMBEDDINGGEMMA_ONNX,
        )
        model_path = hf_hub_download(
            _EMBEDDINGGEMMA_REPO, subfolder="onnx", filename=_EMBEDDINGGEMMA_ONNX
        )
        tok_path = hf_hub_download(_EMBEDDINGGEMMA_REPO, filename="tokenizer.json")

        self._session = ort.InferenceSession(model_path, providers=self._providers)
        out_names = [o.name for o in self._session.get_outputs()]
        self._output_idx = (
            out_names.index("sentence_embedding")
            if "sentence_embedding" in out_names
            else 1
        )

        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_padding()
        tokenizer.enable_truncation(max_length=_EMBEDDINGGEMMA_MAX_LEN)
        self._tokenizer = tokenizer
        self._np = np
        logger.info(
            "EmbeddinggemmaONNX loaded: providers=%s dim=%d",
            self._providers, _EMBEDDINGGEMMA_DIM,
        )
