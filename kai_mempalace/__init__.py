"""Kai MemPalace v3.3.6 — FAISS-powered memory palace with hybrid search, entity graph, and MCP."""

__version__ = "3.3.6"

from kai_mempalace.palace import Palace, SearchResult, mine_lock, MineAlreadyRunning
from kai_mempalace.backends.embedder import NumpyEmbedder, get_embedder
from kai_mempalace.backends.faiss_store import FaissStore
from kai_mempalace.backends.knowledge_graph import KnowledgeGraph

from kai_mempalace.dialect import (
    aaak_compress,
    aaak_decompress,
    aaak_parse_entry,
    aaak_validate,
)
from kai_mempalace.layers import MemoryStack
from kai_mempalace.entity_detector import EntityDetector
from kai_mempalace.entity_registry import EntityRegistry
from kai_mempalace.miner import (
    mine_file_into_palace,
    mine_text_into_palace,
    mine_conversation,
    batch_mine,
    mine_code_file,
    FileMiner,
)
from kai_mempalace.config import KaiPalaceConfig, sanitize_name, sanitize_content
from kai_mempalace.query_sanitizer import sanitize_query
from kai_mempalace.normalize import normalize, strip_noise
from kai_mempalace.exporter import export_palace
from kai_mempalace.sync import sync_palace, SyncReport
from kai_mempalace.dedup import dedup_palace, show_stats
from kai_mempalace.palace_graph import (
    build_graph,
    traverse,
    find_tunnels,
    create_tunnel,
    list_tunnels,
    delete_tunnel,
    follow_tunnels,
    graph_stats,
)
from kai_mempalace.hallways import compute_hallways_for_wing
from kai_mempalace.dynamics import (
    initialize_dynamics_fields, potentiate, apply_decay,
)
from kai_mempalace.general_extractor import extract_memories
from kai_mempalace.room_detector_local import detect_rooms_local
from kai_mempalace.fact_checker import check_text
from kai_mempalace.onboarding import run_onboarding, quick_setup
from kai_mempalace.project_scanner import ProjectInfo, scan
from kai_mempalace.convo_scanner import scan_claude_projects, is_claude_projects_root

__all__ = [
    "Palace",
    "SearchResult",
    "mine_lock",
    "MineAlreadyRunning",
    "NumpyEmbedder",
    "get_embedder",
    "FaissStore",
    "KnowledgeGraph",
    "MemoryStack",
    "EntityDetector",
    "EntityRegistry",
    "aaak_compress",
    "aaak_decompress",
    "aaak_parse_entry",
    "aaak_validate",
    "mine_file_into_palace",
    "mine_text_into_palace",
    "mine_conversation",
    "batch_mine",
    "mine_code_file",
    "FileMiner",
    "KaiPalaceConfig",
    "sanitize_name",
    "sanitize_content",
    "sanitize_query",
    "normalize",
    "strip_noise",
    "export_palace",
    "sync_palace",
    "SyncReport",
    "dedup_palace",
    "show_stats",
    "build_graph",
    "traverse",
    "find_tunnels",
    "create_tunnel",
    "list_tunnels",
    "delete_tunnel",
    "follow_tunnels",
    "graph_stats",
    "compute_hallways_for_wing",
    "initialize_dynamics_fields",
    "potentiate",
    "apply_decay",
    "extract_memories",
    "detect_rooms_local",
    "check_text",
    "run_onboarding",
    "quick_setup",
    "ProjectInfo",
    "scan",
    "scan_claude_projects",
    "is_claude_projects_root",
]
