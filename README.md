# Kai MemPalace

> Fork of [MemPalace](https://github.com/mempalace/mempalace) — local-first AI memory system.

Replaces the ChromaDB/onnxruntime backend with **FAISS + numpy TF-IDF/SVD embeddings**, making it compatible with platforms that don't support onnxruntime (e.g. Alpine Linux on aarch64).

## Key differences from upstream

| Feature | Upstream MemPalace | Kai MemPalace |
|---|---|---|
| Vector store | ChromaDB | FAISS (IndexFlatIP) |
| Embeddings | ONNX MiniLM-L6-v2 (384-dim) | TF-IDF → scipy TruncatedSVD (384-dim) |
| Dependencies | onnxruntime, chromadb | faiss-cpu, numpy, scipy only |
| Platform | glibc Linux, macOS, Windows | Alpine, aarch64, any musl-based |
| API calls | Zero (local) | Zero (local) |
| Vector persistence | ChromaDB managed | Raw vectors stored as SQLite blobs |

## Architecture

```
kai_mempalace/
├── __init__.py          # Package init
├── cli.py               # Full CLI (24 commands)
├── embedder.py          # TF-IDF → SVD → 384-dim vectors (numpy/scipy)
├── faiss_store.py       # FAISS vector index + SQLite metadata
├── knowledge_graph.py   # Temporal entity-relationship store (SQLite)
└── palace.py            # Wings/rooms/drawers manager
```

## Quick start

```bash
# Install deps
pip install faiss-cpu numpy scipy pyyaml python-dateutil

# Use the CLI
python cli.py init
python cli.py add --wing zeth --room preferences --content "Zeth prefers dark mode"
python cli.py search "dark mode"
python cli.py status
```

## CLI commands

| Command | Description |
|---|---|
| `init` | Initialize a new palace |
| `status` | Show wings, rooms, drawer counts |
| `add` | Store a memory drawer |
| `search` | Semantic search across all memories |
| `get` | Get a drawer by ID |
| `list` | List drawers (filter by wing/room) |
| `wings` | List all wings |
| `rooms` | List all rooms |
| `delete` | Delete a drawer |
| `kg-add` | Add a knowledge graph fact |
| `kg-query` | Query facts about an entity |
| `kg-invalidate` | Mark a fact as no longer true |
| `diary` | Write an agent diary entry |
| `diary-read` | Read recent diary entries |
| `check-dup` | Check for duplicate/similar content |

## License

MIT — same as upstream MemPalace.
