"""Numpy-only BERT embedder — pure numpy transformer inference.

No PyTorch, no ONNX Runtime, no TensorFlow. Works on any platform
including Alpine Linux (musl) where PyTorch wheels don't exist.

Downloads MiniLM-L6-v2 weights from HuggingFace as safetensors on
first use, converts to .npz, and caches locally.
"""

import json
import logging
import struct
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_DIM = 384
_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_HF_BASE = "https://huggingface.co"
_CACHE_DIR = Path.home() / ".cache" / "kai-mempalace" / "models" / "all-MiniLM-L6-v2"
_WEIGHTS_FILE = "weights.npz"
_TOKENIZER_FILE = "tokenizer.json"
_CONFIG_FILE = "config.json"

_FILES = {
    "model.safetensors": f"{_HF_BASE}/{_MODEL_ID}/resolve/main/model.safetensors",
    "tokenizer.json": f"{_HF_BASE}/{_MODEL_ID}/resolve/main/tokenizer.json",
    "config.json": f"{_HF_BASE}/{_MODEL_ID}/resolve/main/config.json",
}

_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float32,
    "I64": np.int64,
    "I32": np.int32,
    "I8": np.int8,
}


def _load_safetensors(path: str) -> dict[str, np.ndarray]:
    """Load a safetensors file using only numpy + struct (no torch dep)."""
    with open(path, "rb") as f:
        header_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_bytes)[0]
        header = json.loads(f.read(header_size))
        tensors: dict[str, np.ndarray] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dt = _DTYPE_MAP.get(info["dtype"], np.float32)
            shape = info["shape"]
            begin, end = info["data_offsets"]
            f.seek(8 + header_size + begin)
            data = f.read(end - begin)
            tensors[name] = np.frombuffer(data, dtype=dt).reshape(shape)
        return tensors


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / exp.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class NumpyBertEmbedder:
    """384-dim BERT embedding via pure numpy inference.

    Downloads ``all-MiniLM-L6-v2`` weights from HuggingFace on first
    ``embed()`` call (cached to ``~/.cache/kai-mempalace/``).

    Uses only ``numpy`` + ``tokenizers`` at runtime — both have musl
    wheels, so this works on Alpine Linux out of the box.

    Usage::

        ef = NumpyBertEmbedder()
        vectors = ef.embed(["hello world"])  # -> np.ndarray (n, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._weights: dict[str, np.ndarray] = {}
        self._tokenizer = None
        self._loaded = False
        self._config: dict = {}

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def is_fitted(self) -> bool:
        return self._loaded

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns float32 array shape (n, 384)."""
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        self._lazy_load()
        encs = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encs) if encs else 1
        max_len = min(max_len, 512)
        B = len(texts)
        input_ids = np.zeros((B, max_len), dtype=np.int64)
        attention_mask = np.zeros((B, max_len), dtype=np.int64)
        for i, e in enumerate(encs):
            ids = e.ids[:max_len]
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(ids)] = 1
        return self._forward(input_ids, attention_mask)

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        weights_path = _CACHE_DIR / _WEIGHTS_FILE
        tokenizer_path = _CACHE_DIR / _TOKENIZER_FILE
        config_path = _CACHE_DIR / _CONFIG_FILE
        if not weights_path.exists():
            self._download_and_convert_weights()
        if not tokenizer_path.exists():
            self._download_file(_TOKENIZER_FILE)
        if not config_path.exists():
            self._download_file(_CONFIG_FILE)
        self._weights = dict(np.load(str(weights_path)))
        with open(config_path) as f:
            self._config = json.load(f)
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=512)
        self._loaded = True
        logger.info(
            "NumpyBertEmbedder loaded: model=%s weights=%.1fMB",
            self._model_name,
            weights_path.stat().st_size / 1e6,
        )

    def _download_file(self, filename: str) -> None:
        url = _FILES.get(filename)
        if url is None:
            raise ValueError(f"Unknown file: {filename}")
        dest = _CACHE_DIR / filename
        if dest.exists():
            return
        logger.info("Downloading %s from HuggingFace...", filename)
        tmp = str(dest) + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp)
            Path(tmp).rename(dest)
        except Exception as e:
            Path(tmp).unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {url}: {e}") from e
        logger.info("Downloaded %s (%.1f MB)", filename, dest.stat().st_size / 1e6)

    def _download_and_convert_weights(self) -> None:
        """Download safetensors weights and convert to compressed .npz."""
        safetensors_path = _CACHE_DIR / "model.safetensors"
        if not safetensors_path.exists():
            self._download_file("model.safetensors")
        logger.info("Converting safetensors to numpy...")
        weights = _load_safetensors(str(safetensors_path))
        np.savez_compressed(str(_CACHE_DIR / _WEIGHTS_FILE), **weights)
        logger.info(
            "Converted %d tensors to %s (%.1f MB)",
            len(weights), _WEIGHTS_FILE,
            (_CACHE_DIR / _WEIGHTS_FILE).stat().st_size / 1e6,
        )

    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        w = self._weights
        cfg = self._config
        hidden = cfg.get("hidden_size", 384)
        num_layers = cfg.get("num_hidden_layers", 6)
        num_heads = cfg.get("num_attention_heads", 6)
        head_dim = hidden // num_heads
        eps = cfg.get("layer_norm_eps", 1e-12)
        B, S = input_ids.shape
        # ── Embeddings ────────────────────────────────────────────────
        word_emb = w["embeddings.word_embeddings.weight"]
        pos_emb = w["embeddings.position_embeddings.weight"]
        tok_emb = w["embeddings.token_type_embeddings.weight"]
        ln_w = w["embeddings.LayerNorm.weight"]
        ln_b = w["embeddings.LayerNorm.bias"]
        positions = np.arange(S, dtype=np.int64)[None, :]
        h = (
            word_emb[input_ids]
            + pos_emb[positions]
            + tok_emb[np.zeros((B, S), dtype=np.int64)]
        )
        h = _layer_norm(h, ln_w, ln_b, eps)
        # ── Transformer layers ────────────────────────────────────────
        for i in range(num_layers):
            p = f"encoder.layer.{i}."
            Q = h @ w[p + "attention.self.query.weight"].T + w[p + "attention.self.query.bias"]
            K = h @ w[p + "attention.self.key.weight"].T + w[p + "attention.self.key.bias"]
            V = h @ w[p + "attention.self.value.weight"].T + w[p + "attention.self.value.bias"]
            # Reshape for multi-head: (B, S, num_heads, head_dim)
            Q = Q.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
            K = K.reshape(B, S, num_heads, head_dim).transpose(0, 2, 3, 1)
            V = V.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
            scores = Q @ K / (head_dim ** 0.5)
            mask = attention_mask[:, None, None, :]
            scores = scores * mask + (1 - mask) * (-1e9)
            attn_out = _softmax(scores, axis=-1) @ V
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, hidden)
            attn_out = attn_out @ w[p + "attention.output.dense.weight"].T + w[p + "attention.output.dense.bias"]
            h = _layer_norm(h + attn_out, w[p + "attention.output.LayerNorm.weight"],
                            w[p + "attention.output.LayerNorm.bias"], eps)
            # Feed-forward
            ffn = _gelu(h @ w[p + "intermediate.dense.weight"].T + w[p + "intermediate.dense.bias"])
            ffn = ffn @ w[p + "output.dense.weight"].T + w[p + "output.dense.bias"]
            h = _layer_norm(h + ffn, w[p + "output.LayerNorm.weight"],
                            w[p + "output.LayerNorm.bias"], eps)
        # ── Mean pooling + L2 normalize ───────────────────────────────
        mask_3d = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_emb = np.sum(h * mask_3d, axis=1)
        sum_mask = np.sum(mask_3d, axis=1)
        pooled = sum_emb / np.clip(sum_mask, a_min=1e-9, a_max=None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
        return (pooled / norms).astype(np.float32)
