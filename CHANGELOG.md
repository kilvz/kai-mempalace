# Changelog

## v4.0.0 (2026-05-29)

Major restructuring and feature expansion. The codebase has been refactored from flat root-level files into a proper `kai_mempalace/` subpackage, and 6 major features from upstream MemPalace have been ported.

### New features

- **FTS5 proximity search** — `_build_fts_query()` adds `*` prefix wildcards to bare terms and passes through AND/OR/NOT/NEAR/N operators verbatim
- **Wikipedia entity lookup** — `entity_registry.py`: `research()` + `confirm_research()` methods with opt-in network lookup via Wikipedia REST API, KG-cached
- **Interactive entity confirmation** — `entity_detector.py`: `confirm_entities()` interactive prompt with confirm/reject/rename flow
- **Project scanner enrichment** — `project_scanner.py`: `PersonInfo` dataclass with `confidence`/`to_signal()`, git author analysis via `_git_authors()`, bot filtering (`_is_bot()`), `_looks_like_real_name()` heuristic, `ProjectInfo.to_signal()`
- **Sweeper TTL config** — `sweeper.py`: `skip_before`, `exclude_patterns`, `dry_run` parameters on `sweep()`
- **Closet-boosted reranking** — `palace.py`: `_get_closet_source_ids()` + +0.15 rank boost in `_hybrid_search()`

### Breaking changes

- **Package restructured** — all modules moved from repo root to `kai_mempalace/` subpackage. Root files are now shims that re-export from the subpackage.
- **`scan()` return type changed** — second element is now `list[PersonInfo]` instead of `list[ProjectInfo]`
- **`sweep()` return dict** — now includes `drawers_excluded` and `dry_run` keys
- **`research()` default** — `allow_network` defaults to `False` (privacy-first); caller must explicitly opt in

### Fixes

- `_wikipedia_lookup()` now uses Wikipedia REST API (`/api/rest_v1/page/summary/`) instead of the Action API, matching upstream
- `research()` return dict aligned with upstream: `inferred_type`, `confidence`, `wiki_summary`, `wiki_title`
- KG caching now serializes full wiki lookup result as JSON in the `object` field (avoids unsupported kwargs)
- `ProjectInfo.confidence` values aligned with upstream (0.7 for git-only, 0.85 for manifest-only)

## v1.0.0 (2026-05-25)

Initial release of **Kai MemPalace** — a fork of [MemPalace](https://github.com/mempalace/mempalace) with a custom FAISS-powered backend.

### What's different from upstream

- **Replaced ChromaDB + onnxruntime** with a pure FAISS vector store — runs on Alpine Linux aarch64 and other platforms without onnxruntime support
- **Custom embedding pipeline** using numpy/scipy TF-IDF → TruncatedSVD (384-dim) instead of ONNX MiniLM-L6-v2
- **Vector persistence** — raw embedding vectors are stored as SQLite blobs so the FAISS index can be rebuilt after deletions without losing data
- **ID collision protection** — sequence counter falls back to max existing doc ID if the counter file is missing
- **Minimal dependencies** — only faiss-cpu, numpy, scipy, pyyaml, python-dateutil
- **Zero API calls** — fully local, no cloud dependencies

### Features

- Semantic search across all memories (wings/rooms/drawers)
- Temporal knowledge graph with validity windows
- Agent diary system
- Configurable embedding pipeline
- Full CLI with 24 commands
- Compatible with MemPalace concepts and workflows
