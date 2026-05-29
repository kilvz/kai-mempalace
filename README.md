# Kai MemPalace

> Local-first AI memory system. Fork of [MemPalace](https://github.com/mempalace/mempalace).

Replaces ChromaDB/onnxruntime with **FAISS + numpy TF-IDF/SVD embeddings** — runs on Alpine Linux aarch64 and any musl-based platform.

## Key differences from upstream

| Feature | Upstream MemPalace | Kai MemPalace |
|---|---|---|
| Vector store | ChromaDB | FAISS (IndexFlatIP) |
| Embeddings | ONNX MiniLM-L6-v2 (384-dim) | TF-IDF → scipy TruncatedSVD (384-dim) |
| Dependencies | onnxruntime, chromadb | faiss-cpu, numpy, scipy only |
| Platform | glibc Linux, macOS, Windows | Alpine, aarch64, any musl-based |
| API calls | Zero (local) | Zero (local) |
| Vector persistence | ChromaDB managed | Raw vectors stored as SQLite blobs |
| Entity registry | JSON-file backed | KnowledgeGraph backed |
| Wiki lookup | Wikipedia REST API (opt-in) | Wikipedia REST API (opt-in), KG-cached |
| Search modes | vector, keyword, hybrid | vector, keyword (FTS5), hybrid + closet boost |
| Project scanner | Manifest + git author analysis | Same, plus bot filtering, PersonInfo dataclass |

## Architecture

```
kai_mempalace/
├── __init__.py             # Package init & re-exports
├── backends/
│   ├── embedder.py         # TF-IDF → SVD → 384-dim vectors
│   ├── faiss_store.py      # FAISS index + SQLite metadata
│   ├── knowledge_graph.py  # Temporal entity-relationship store
│   ├── registry.py         # Backend registry
│   └── types.py            # Shared types
├── palace.py               # Wings/rooms/drawers manager (core)
├── palace_graph.py         # Cross-wing tunnel computation
├── sweeper.py              # Message-granular session miner
├── entity_registry.py      # KG-backed entity registry + wiki lookup
├── entity_detector.py      # Regex + heuristic NER
├── project_scanner.py      # Manifest + git history analysis
├── searcher.py             # Search routing & reranking
├── cli.py                  # Full CLI (24+ commands)
├── convo_miner.py          # Conversation transcript mining
├── llm_refine.py           # LLM-based entity refinement
├── miner.py                # File-level memory mining
├── sources/                # Source file abstraction layer
├── instructions/           # Help text for CLI
└── i18n/                   # Internationalization (10+ locales)
```

## Quick start

```bash
pip install kai-mempalace

kai-mempalace init
kai-mempalace add --wing zeth --room preferences --content "Zeth prefers dark mode"
kai-mempalace search "dark mode"
kai-mempalace status
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
| `kg-query` | Query the knowledge graph |
| `kg-invalidate` | Mark a fact as no longer true |
| `diary` | Write an agent diary entry |
| `diary-read` | Read recent diary entries |
| `check-dup` | Check for duplicate/similar content |
| `export` | Export drawers to JSON |
| `sweep` | Ingest session transcripts |
| `scan` | Detect projects and people from codebase |

## New in v4.0.0

- **FTS5 proximity search** — prefix wildcards, AND/OR/NOT/NEAR operators
- **Wikipedia entity lookup** — opt-in network research, KG-cached
- **Interactive entity confirmation** — review/accept/reject/rename detected entities
- **Project scanner enrichment** — `PersonInfo` dataclass, git author extraction, bot filtering
- **Sweeper TTL config** — `skip_before`, `exclude_patterns`, `dry_run` modes
- **Closet-boosted reranking** — results from files referenced in closets get +0.15 rank boost
- **Module restructuring** — flat root files refactored into `kai_mempalace/` subpackage

## License

MIT — same as upstream MemPalace.
