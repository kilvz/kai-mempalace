"""Backend implementations: FAISS vector store, TF-IDF embedder, SQLite knowledge graph."""

from kai_mempalace.backends.embedder import (
    NumpyEmbedder, SentenceTransformerEmbedder, SpacyGloveEmbedder, get_embedder,
)
from kai_mempalace.backends.faiss_store import FaissStore
from kai_mempalace.backends.knowledge_graph import KnowledgeGraph

__all__ = [
    "NumpyEmbedder",
    "SentenceTransformerEmbedder",
    "SpacyGloveEmbedder",
    "get_embedder",
    "FaissStore",
    "KnowledgeGraph",
]
