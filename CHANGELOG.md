# Changelog

## v1.0.0 (2026-05-25)

Initial release of **Kai MemPalace** — a fork of [MemPalace](https://github.com/mempalace/mempalace) with a custom FAISS-powered backend.

### What's different from upstream

- **Replaced ChromaDB + onnxruntime** with a pure FAISS vector store — runs on Alpine Linux aarch64 and other platforms without onnxruntime support
- **Custom embedding pipeline** using numpy/scipy TF-IDF → TruncatedSVD (384-dim) instead of ONNX MiniLM-L6-v2
- **Vector persistence** — raw embedding vectors are stored as SQLite blobs so the FAISS index can be rebuilt after deletions without losing data
- **ID collision protection** — sequence counter falls back to max existing doc ID if the counter file is missing
- **Minimal dependencies** — only faiss-cpu, numpy, scipy, pyyaml, python-dateutil
- **Zero API calls** — fully local, no cloud dependencies

### Architecture

```
kai_mempalace/
├── embedder.py          # TF-IDF → SVD → 384-dim vectors
├── faiss_store.py       # FAISS vector index + SQLite metadata
├── knowledge_graph.py   # Temporal entity-relationship store
├── palace.py            # Wings/rooms/drawers manager
└── cli.py               # Full CLI (24 commands)
```

### Features

- Semantic search across all memories (wings/rooms/drawers)
- Temporal knowledge graph with validity windows
- Agent diary system
- Configurable embedding pipeline
- Full CLI with 24 commands
- Compatible with MemPalace concepts and workflows

### Commands

`init`, `status`, `add`, `search`, `get`, `list`, `wings`, `rooms`, `delete`,
`kg-add`, `kg-query`, `kg-invalidate`, `diary`, `diary-read`, `check-dup`
