"""Kai MemPalace — FAISS-powered memory palace for Kai 9000."""
from embedder import NumpyEmbedder, get_embedder
from faiss_store import FaissStore
from knowledge_graph import KnowledgeGraph
from palace import Palace, SearchResult
