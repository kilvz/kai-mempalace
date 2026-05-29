# Kai MemPalace

> Local-first AI memory system. Fork of [MemPalace](https://github.com/mempalace/mempalace). **v4.2.0 — 40 MCP tools**, FAISS backend, musl-compatible.

Replaces ChromaDB/onnxruntime with **FAISS + pluggable embeddings** — runs on Alpine Linux aarch64 and any musl-based platform.

---

## Comprehensive: Differences from upstream MemPalace

### Architectural

| Layer | Upstream MemPalace (v3.3.6) | Kai MemPalace (v4.2.0) |
|---|---|---|
| **Vector store** | ChromaDB (PersistentClient, HNSW index) | FAISS (IndexFlatIP, brute-force IP) |
| **Vector persistence** | ChromaDB-managed (SQLite + HNSW segment files) | Raw numpy vectors stored as SQLite blobs; FAISS index rebuilt on load |
| **Embedding pipeline** | ONNX `all-MiniLM-L6-v2` (384d) or `embeddinggemma-300m` (multilingual) | **Pluggable**: SentenceTransformer (MiniLM, 384d), spaCy GloVe (300→384d), or numpy TF-IDF+TruncatedSVD (384d) |
| **Embedder switching** | Config-only, requires backend swap | `set_embedder()` MCP tool — hot-switch at runtime, auto-reindex all drawers |
| **Default embedder** | ONNX MiniLM-L6-v2 (requires onnxruntime) | SentenceTransformer MiniLM-L6-v2 (auto-falls back to numpy if missing) |
| **Auto-fallback** | None — crashes if ONNX model unavailable | `_resolve_embedder()` — missing dep → logs warning → uses numpy |
| **AI-settable default** | Config file only | `set_default_embedder` MCP tool — persists to `~/.kai-palace/config.json` |
| **Backend abstraction** | Formal `BaseBackend`/`BaseCollection` ABCs with `PalaceRef`, `QueryResult`, `HealthStatus`, `BackendError` hierarchy | Simplified `base.py` — minimal ABC, direct `FaissBackend` usage |
| **Search engine** | ChromaDB query + BM25 fallback; temporal boosting pipeline | Vector (FAISS), keyword (FTS5), hybrid + closet-boosted reranking + neighbor expansion |
| **Search strategies** | Fixed pipeline | `"vector"` / `"union"` candidate modes via `_CANDIDATE_MERGERS` dispatch |
| **Hardware acceleration** | CUDA, CoreML, DirectML via ONNX Runtime providers | CPU-only (FAISS CPU) |
| **Schema versioning** | No explicit version tracking | Version-tracked schema (`migrate` + FAISS rebuild commands) |
| **Entity registry** | JSON-file backed | KnowledgeGraph-backed |
| **Tunnel/graph system** | Static relationship file | Compute-on-`hallways()` co-occurrence + dynamics fields (strength/stability/access_count) |

### Files

| Upstream has | Kai counterpart or replacement |
|---|---|
| `mempalace/backends/chroma.py` (64 KB) | `backends/faiss_store.py` + `backends/faiss_backend.py` |
| `mempalace/embedding.py` (10.7 KB) | `backends/embedder.py` (TF-IDF+SVD, plus SentenceTransformer/Spacy wrappers) |
| `mempalace/backends/base.py` (12.5 KB, formal ABCs) | `backends/base.py` (minimal) |

| Kai has | What it does |
|---|---|
| `backends/embedder.py` | `SentenceTransformerEmbedder`, `SpacyGloveEmbedder`, `NumpyEmbedder` — pluggable via `get_embedder()` factory |
| `backends/faiss_store.py` | FAISS index + SQLite blob persistence |
| `backends/knowledge_graph.py` | Temporal entity-relationship store (upstream has this at root level) |
| `sources/` (4 files) | Source file abstraction: base, context, registry, transforms |
| `instructions/` (5 `.md`) | Help text for CLI commands |
| `hallways.py` | Entity co-occurrence hallway computation |
| `dynamics.py` | Hebbian potentiation + Ebbinghaus decay fields |
| `llm_refine.py` | LLM-based entity refinement |
| `closet_llm.py` | LLM-powered closet analysis |

### Dependencies

| Upstream MemPalace | Kai MemPalace |
|---|---|
| `chromadb>=1.5.4,<2` | **removed** — replaced by `faiss-cpu` |
| `onnxruntime` (or `-gpu`/`-directml`) | **removed** — no ONNX dependency |
| `huggingface_hub>=0.20` | **removed** |
| `tokenizers>=0.15` | **removed** |
| `pyyaml>=6.0` | **removed** |
| `tomli` (python <3.11) | **removed** |
| `python-dateutil>=2.8` | **removed** (optional fallback) |
| `faiss-cpu>=1.7.0` | **added** |
| `scipy>=1.10.0` | **added** |
| `sentence-transformers` | **optional** (~1.5GB with PyTorch) |
| `spacy` | **optional** (`en_core_web_md`) |

### Platform support

| Platform | Upstream | Kai |
|---|---|---|
| glibc Linux (x86_64) | Yes | Yes |
| glibc Linux (aarch64) | Yes | Yes |
| **Alpine Linux (musl, aarch64)** | **No** (chromadb/onnxruntime not available) | **Yes** |
| macOS (Intel + Apple Silicon) | Yes | Yes |
| Windows (x86_64) | Yes | Yes |
| Any musl-based distro | **No** | **Yes** |

The fork exists because upstream's ChromaDB dependency requires `grpcio`, which fails to compile on musl. Kai uses FAISS + numpy — both available on musl.

### MCP tools

**Upstream:** 30 tools (`mempalace_*` prefix)  
**Kai:** 40 tools (short names, no prefix)

| Category | Tools | Kai vs Upstream |
|---|---|---|
| **Status/info** | `get_status`, `list_wings`, `list_rooms`, `get_taxonomy`, `memories_filed_away`, `get_aaak_spec` | **Same** |
| **Search** | `search`, `check_duplicate` | **Same** (Kai adds FTS5 + closet boost) |
| **Drawers** | `add_drawer`, `get_drawer`, `list_drawers`, `update_drawer`, `delete_drawer` | **Same** |
| **Knowledge graph** | `kg_add`, `kg_query`, `kg_invalidate`, `kg_stats`, `kg_timeline` | **Same** |
| **Diary** | `diary_write`, `diary_read` | **Same** |
| **Tunnels/graph** | `create_tunnel`, `delete_tunnel`, `find_tunnels`, `follow_tunnels`, `list_tunnels`, `traverse`, `graph_stats` | **Same** |
| **Management** | `sync`, `reconnect`, `hook_settings`, `rebuild_fts` | **Same** |
| **Mining** | `mine_text`, `mine_file`, `batch_mine` | **Kai only** (upstream mines via CLI only) |
| **AAAK** | `aaak_compress`, `aaak_decompress`, `aaak_parse` | **Kai only** |
| **Embedders** | `set_embedder`, `get_default_embedder`, `set_default_embedder` | **Kai only** |

### 10 MCP tools exclusive to Kai

| Tool | Description |
|---|---|
| `mine_text` | Mine raw text into palace on-the-fly (no file needed) |
| `mine_file` | Mine a file into palace |
| `batch_mine` | Batch-mine all matching files in a directory |
| `aaak_compress` | Compress text to AAAK dialect |
| `aaak_decompress` | Decompress AAAK to readable text |
| `aaak_parse` | Parse an AAAK entry into structured fields |
| `rebuild_fts` | Rebuild the FTS5 full-text search index |
| `set_embedder` | Hot-switch the active embedder at runtime |
| `get_default_embedder` | Get the global default embedder for new palaces |
| `set_default_embedder` | Set the global default embedder (persisted) |

---

## Architecture

```
kai_mempalace/
├── __init__.py              # Package init & re-exports (v4.2.0, 40 MCP tools)
├── backends/
│   ├── embedder.py          # Pluggable: TF-IDF+SVD, SentenceTransformer, spaCy GloVe
│   ├── faiss_store.py       # FAISS index + SQLite blob metadata
│   ├── knowledge_graph.py   # Temporal entity-relationship store
│   ├── registry.py          # Backend registry
│   ├── base.py              # Minimal base ABC
│   └── types.py             # Shared type definitions
├── palace.py                 # Wings/rooms/drawers manager (core)
├── palace_graph.py           # Cross-wing tunnel computation
├── sweeper.py                # Message-granular session miner
├── entity_registry.py        # KG-backed entity registry + wiki lookup
├── entity_detector.py        # Regex + heuristic NER, confirm_entities()
├── project_scanner.py        # Manifest + git history + discover_entities()
├── searcher.py               # FTS5, BM25, strategy system, neighbor expansion
├── cli.py                    # Full CLI (24+ commands)
├── convo_miner.py            # Conversation transcript mining
├── llm_refine.py             # LLM-based entity refinement
├── miner.py                  # File-level + entity-line mining
├── mcp_server.py             # MCP protocol server (40 tools)
├── migrate.py                # Schema migration + FAISS rebuild
├── sync.py                   # Closet cleanup + palace sync
├── hallways.py               # Entity co-occurrence hallways
├── dynamics.py               # Hebbian potentiation + Ebbinghaus decay
├── sources/                  # Source file abstraction layer
├── instructions/             # Help text for CLI
└── i18n/                     # Internationalization (14 locales)
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

## New in v4.2.0

- **Pluggable embedders** — `SentenceTransformerEmbedder` (MiniLM, 384d), `SpacyGloveEmbedder` (GloVe 300→384d via zero-pad), auto-fallback
- **AI-settable default embedder** — `set_default_embedder`/`get_default_embedder` MCP tools; no server restart needed
- **Auto-fallback** — missing `sentence-transformers`/`spacy`/`onnxruntime` → logs warning → uses `numpy` silently
- **2 new MCP tools** — kai-palace now has **40** total MCP tools (up from 38)

## v4.1.0 features

- **MCP protocol server** — stdio transport, 35 tools (up from 18)
- **17 ported MCP tools** — taxonomy, graph stats, tunnels, sync, hook settings, rebuild FTS, update drawer, check duplicate, KG timeline, reconnect, AAAK spec
- **Schema migration** — `migrate` command with version-tracked FAISS rebuild
- **Content-date extraction** — 5-fallback hierarchy, auto-tagged on mine
- **Project scanner** — `PersonInfo`/`ProjectInfo` dataclasses, `discover_entities()`, git author + manifest analysis
- **Search strategy system** — `"vector"` / `"union"` candidate modes, neighbor expansion
- **Entity-line mining** — per-line entity annotations alongside drawers
- **Hallway dynamics** — `initialize_dynamics_fields` wired (strength/stability/access_count)
- **Closet cleanup** — orphaned closets purged on sync

## v4.0.0 features

- FTS5 proximity search — prefix wildcards, AND/OR/NOT/NEAR operators
- Wikipedia entity lookup — opt-in network research, KG-cached
- Interactive entity confirmation — review/accept/reject/rename detected entities
- Closet-boosted reranking — +0.15 rank boost from closet hits
- Module restructuring — flat root files refactored into `kai_mempalace/` subpackage

## License

MIT — same as upstream MemPalace.
