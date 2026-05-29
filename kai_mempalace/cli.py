"""Kai MemPalace CLI — manage your memory palace from the command line."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import kai_mempalace
from kai_mempalace.palace import Palace

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Kai MemPalace CLI")
    parser.add_argument("--palace", default="~/.kai-palace",
                        help="Path to the palace directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Initialize palace")
    p_init.add_argument("--version", type=int, default=2)

    sub.add_parser("status", help="Show palace status")

    p_add = sub.add_parser("add", help="Add a drawer")
    p_add.add_argument("--wing", "-w", required=True)
    p_add.add_argument("--room", "-r", required=True)
    p_add.add_argument("--content", "-c", required=True)
    p_add.add_argument("--meta", "-m", default="{}")
    p_add.add_argument("--source", "-s", default="")

    p_srch = sub.add_parser("search", help="Search the palace")
    p_srch.add_argument("query", nargs="?", default="")
    p_srch.add_argument("--wing", "-w")
    p_srch.add_argument("--room", "-r")
    p_srch.add_argument("--limit", "-l", type=int, default=10)
    p_srch.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")

    p_get = sub.add_parser("get", help="Get a drawer by ID")
    p_get.add_argument("drawer_id")

    p_list = sub.add_parser("list", help="List drawers")
    p_list.add_argument("--wing", "-w")
    p_list.add_argument("--room", "-r")
    p_list.add_argument("--limit", "-l", type=int, default=20)
    p_list.add_argument("--offset", "-o", type=int, default=0)

    p_wing = sub.add_parser("wings", help="List wings")
    p_wing.add_argument("--verbose", "-v", action="store_true")

    p_room = sub.add_parser("rooms", help="List rooms")
    p_room.add_argument("--wing", "-w")

    p_del = sub.add_parser("delete", help="Delete a drawer")
    p_del.add_argument("drawer_id")

    p_wing_del = sub.add_parser("delete-wing", help="Delete a wing")
    p_wing_del.add_argument("name")

    p_room_del = sub.add_parser("delete-room", help="Delete a room")
    p_room_del.add_argument("--wing", "-w", required=True)
    p_room_del.add_argument("--room", "-r", required=True)

    p_kga = sub.add_parser("kg-add", help="Add KG fact")
    p_kga.add_argument("--subject", "-s", required=True)
    p_kga.add_argument("--predicate", "-p", required=True)
    p_kga.add_argument("--object", "-o", required=True)
    p_kga.add_argument("--source", default="")
    p_kga.add_argument("--valid-from")

    p_kgq = sub.add_parser("kg-query", help="Query KG")
    p_kgq.add_argument("entity", nargs="?", default="")
    p_kgq.add_argument("--predicate")
    p_kgq.add_argument("--as-of")
    p_kgq.add_argument("--direction", default="both",
                       choices=["outgoing", "incoming", "both"])
    p_kgq.add_argument("--all", "-a", action="store_true")

    p_kgi = sub.add_parser("kg-invalidate", help="Invalidate KG fact")
    p_kgi.add_argument("--subject", "-s", required=True)
    p_kgi.add_argument("--predicate", "-p", required=True)
    p_kgi.add_argument("--object", "-o", required=True)
    p_kgi.add_argument("--ended")

    p_kg_stats = sub.add_parser("kg-stats", help="KG statistics")

    p_diary = sub.add_parser("diary", help="Write diary entry")
    p_diary.add_argument("--agent", "-a", required=True)
    p_diary.add_argument("--entry", "-e", required=True)
    p_diary.add_argument("--topic", "-t", default="general")
    p_diary.add_argument("--wing", "-w", default="")

    p_diaryr = sub.add_parser("diary-read", help="Read diary entries")
    p_diaryr.add_argument("--agent", "-a", required=True)
    p_diaryr.add_argument("--last-n", type=int, default=10)

    p_dedup = sub.add_parser("check-dup", help="Check for duplicate content")
    p_dedup.add_argument("content")
    p_dedup.add_argument("--threshold", type=float, default=0.9)

    p_rebuild = sub.add_parser("rebuild-fts", help="Rebuild FTS index")
    p_aaak = sub.add_parser("aaak", help="Compress text to AAAK")
    p_aaak.add_argument("text", nargs="?", default="")
    p_aaak.add_argument("--output-format", choices=["aaak", "json"],
                        default="aaak")

    p_mine = sub.add_parser("mine", help="Mine a file into the palace")
    p_mine.add_argument("file")
    p_mine.add_argument("--wing", "-w", required=True)
    p_mine.add_argument("--room", "-r", required=True)

    p_mcp = sub.add_parser("mcp", help="Run MCP server")
    p_mcp.add_argument("--host", default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=8316)
    p_mcp.add_argument("--transport", choices=["stdio", "sse"], default="stdio")

    # -- Ported upstream commands --

    p_sweep = sub.add_parser("sweep", help="Sweep .jsonl files into palace (message-granular mine)")
    p_sweep.add_argument("path", help="Path to .jsonl file or directory")

    p_sync = sub.add_parser("sync", help="Prune stale drawers (gitignored/deleted sources)")
    p_sync.add_argument("--wing", "-w", help="Limit to wing")
    p_sync.add_argument("--project-dir", "-d", action="append", help="Project root directory")
    p_sync.add_argument("--dry-run", "-n", action="store_true", help="Preview only")
    p_sync.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")

    p_split = sub.add_parser("split", help="Split mega-files into per-session files")
    p_split.add_argument("--source", help="Source directory (default: ~/Desktop/transcripts)")
    p_split.add_argument("--output-dir", help="Output directory")
    p_split.add_argument("--file", help="Single file to split")
    p_split.add_argument("--min-sessions", type=int, default=2)
    p_split.add_argument("--dry-run", action="store_true")

    p_hook = sub.add_parser("hook", help="Run hook logic for Claude Code / Codex")
    hook_sub = p_hook.add_subparsers(dest="hook_command", required=True)
    p_hook_run = hook_sub.add_parser("run", help="Run a hook")
    p_hook_run.add_argument("--hook", required=True, choices=["session-start", "stop", "precompact"])
    p_hook_run.add_argument("--harness", required=True, choices=["claude-code", "codex"])

    p_instr = sub.add_parser("instructions", help="Output skill instructions")
    instr_sub = p_instr.add_subparsers(dest="instr_command", required=True)
    for instr_name in ("init", "search", "mine", "help", "status"):
        instr_sub.add_parser(instr_name, help=f"Instructions for {instr_name}")

    p_migrate = sub.add_parser("migrate", help="Schema migration and FAISS rebuild")
    p_migrate.add_argument("--dry-run", "-n", action="store_true",
                           help="Show pending migrations without applying")
    p_migrate.add_argument("--rebuild-faiss", "-f", action="store_true",
                           help="Rebuild FAISS index from SQLite data")
    p_migrate.add_argument("--status", "-s", action="store_true",
                           help="Show schema version and migration status")

    p_repair = sub.add_parser("repair", help="Repair utilities: integrity, VACUUM, FTS5 rebuild")
    p_repair.add_argument("--integrity", action="store_true", help="Check SQLite integrity")
    p_repair.add_argument("--vacuum", action="store_true", help="Run VACUUM")
    p_repair.add_argument("--rebuild-fts", action="store_true", help="Rebuild FTS5 index")
    p_repair.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    p_repair_status = sub.add_parser("repair-status", help="Palace health check")

    p_wake = sub.add_parser("wake-up", help="Show L0+L1 wake-up context")
    p_wake.add_argument("--agent", required=True, help="Agent name")
    p_wake.add_argument("--last-n", type=int, default=5, help="Entries per layer")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    palace = Palace(args.palace)
    palace.init()

    try:
        if args.command == "init":
            print(f"Palace initialized at {Path(args.palace).expanduser().resolve()}")

        elif args.command == "status":
            s = palace.status()
            print(json.dumps(s, indent=2))

        elif args.command == "add":
            meta = json.loads(args.meta)
            did = palace.add_drawer(args.wing, args.room, args.content, meta, args.source)
            print(f"Added drawer: {did}")

        elif args.command == "search":
            if not args.query:
                args.query = sys.stdin.read().strip()
            results = palace.search(args.query, n_results=args.limit,
                                     wing=args.wing, room=args.room, mode=args.mode)
            if not results:
                print("No results found.")
            else:
                for i, r in enumerate(results):
                    print(f"\n--- Result {i+1} (distance: {r.distance:.4f}) ---")
                    print(f"  ID:   {r.id}")
                    print(f"  Wing: {r.wing} / Room: {r.room}")
                    text = r.text[:300] + ("..." if len(r.text) > 300 else "")
                    print(f"  Text: {text}")

        elif args.command == "get":
            d = palace.get_drawer(args.drawer_id)
            if d:
                print(json.dumps(d, indent=2, default=str))
            else:
                print(f"Drawer {args.drawer_id} not found")

        elif args.command == "list":
            drawers = palace.list_drawers(args.wing, args.room, args.limit, args.offset)
            if not drawers:
                print("No drawers found.")
            else:
                for d in drawers:
                    print(f"  {d['id']:20s} | {d['wing']:15s} / {d['room']:20s} | {d['created_at']}")
                    print(f"  {d['content']}")
                    print()

        elif args.command == "wings":
            wings = palace.list_wings()
            if not wings:
                print("No wings.")
            else:
                for w in wings:
                    desc = f" - {w['description']}" if w['description'] and args.verbose else ""
                    print(f"  {w['name']:20s}  {w['drawer_count']} drawers{desc}")

        elif args.command == "rooms":
            rooms = palace.list_rooms(args.wing)
            if not rooms:
                print("No rooms.")
            else:
                for r in rooms:
                    print(f"  {r['wing']:15s} / {r['name']:20s}  {r['drawer_count']} drawers")

        elif args.command == "delete":
            ok = palace.delete_drawer(args.drawer_id)
            print("Deleted." if ok else "Not found.")

        elif args.command == "delete-wing":
            palace.delete_wing(args.name)
            print(f"Wing {args.name} deleted.")

        elif args.command == "delete-room":
            palace.delete_room(args.wing, args.room)
            print(f"Room {args.wing}/{args.room} deleted.")

        elif args.command == "kg-add":
            fid = palace.kg.add(args.subject, args.predicate, args.object,
                                valid_from=args.valid_from, source=args.source)
            print(f"Added fact #{fid}")

        elif args.command == "kg-query":
            if args.all:
                facts = palace.kg.query(as_of=args.as_of)
            else:
                facts = palace.kg.query(entity=args.entity or None,
                                        predicate=args.predicate,
                                        as_of=args.as_of, direction=args.direction)
            if not facts:
                print("No facts found.")
            else:
                for f in facts:
                    valid = f" [{f['valid_from']} -> {f['valid_to'] or 'now'}]" if f['valid_from'] else ""
                    print(f"  {f['subject']} -- {f['predicate']} -- {f['object']}{valid}")

        elif args.command == "kg-invalidate":
            n = palace.kg.invalidate(args.subject, args.predicate, args.object, args.ended)
            print(f"Invalidated {n} fact(s).")

        elif args.command == "kg-stats":
            s = palace.kg.stats()
            print(json.dumps(s, indent=2))

        elif args.command == "diary":
            wing = palace.diary_write(args.agent, args.entry, args.topic, args.wing)
            print(f"Diary entry written to wing: {wing}")

        elif args.command == "diary-read":
            entries = palace.diary_read(args.agent, args.last_n)
            if not entries:
                print("No diary entries.")
            else:
                for e in entries:
                    print(f"\n[{e['created_at']}] topic={e['metadata'].get('topic', '?')}")
                    print(f"  {e['content'][:200]}")

        elif args.command == "check-dup":
            dup = palace.check_duplicate(args.content, args.threshold)
            if dup:
                print(f"Similar content found (similarity={dup['similarity']:.4f}):")
                print(f"  ID:   {dup['id']}")
                print(f"  Text: {dup['text'][:200]}")
            else:
                print("No duplicate found.")

        elif args.command == "rebuild-fts":
            palace.rebuild_fts()
            print("FTS index rebuilt.")

        elif args.command == "aaak":
            from kai_mempalace.dialect import aaak_compress, aaak_parse_entry
            text = args.text if args.text else sys.stdin.read().strip()
            if args.output_format == "json":
                parsed = aaak_parse_entry(text)
                print(json.dumps(parsed, indent=2))
            else:
                compressed = aaak_compress(text)
                print(compressed)

        elif args.command == "mine":
            from kai_mempalace.miner import mine_file_into_palace
            count = mine_file_into_palace(palace, args.file, args.wing, args.room)
            print(f"Mined {count} items from {args.file}")

        # ── Ported upstream command handlers ─────────────────────────

        elif args.command == "sweep":
            from kai_mempalace.sweeper import sweep, sweep_directory
            p = Path(args.path)
            if p.is_dir():
                result = sweep_directory(str(p), args.palace)
                print(f"Swept directory: {result.get('files_succeeded', 0)} files, "
                      f"{result['total_added']} added, {result['total_skipped']} skipped")
            else:
                result = sweep(str(p), args.palace, source_label=str(p))
                print(f"Swept {p.name}: {result['drawers_added']} added, "
                      f"{result['drawers_already_present']} existing, "
                      f"{result['drawers_skipped']} skipped")

        elif args.command == "sync":
            from kai_mempalace.sync import sync_palace
            dry_run = not args.apply
            project_dirs = args.project_dir or None
            report = sync_palace(
                args.palace, project_dirs=project_dirs,
                wing=args.wing, dry_run=dry_run,
            )
            print(f"Sync {'(dry run)' if dry_run else ''}: "
                  f"{report['scanned']} scanned, "
                  f"{report['kept']} kept, "
                  f"{report['gitignored']} gitignored, "
                  f"{report['missing']} missing, "
                  f"{report['removed_drawers']} removed")

        elif args.command == "split":
            from kai_mempalace.split_mega_files import split_file
            if args.file:
                written = split_file(args.file, args.output_dir, dry_run=args.dry_run)
                print(f"Split {args.file}: {len(written)} sessions")
            else:
                from kai_mempalace.split_mega_files import main as split_main
                # Rebuild argv for split_main's own argument parser
                old_argv = sys.argv
                split_argv = ["split"]
                if args.source:
                    split_argv.extend(["--source", args.source])
                if args.output_dir:
                    split_argv.extend(["--output-dir", args.output_dir])
                if args.dry_run:
                    split_argv.append("--dry-run")
                split_argv.extend(["--min-sessions", str(args.min_sessions)])
                sys.argv = split_argv
                try:
                    split_main()
                finally:
                    sys.argv = old_argv

        elif args.command == "hook":
            if args.hook_command == "run":
                from kai_mempalace.hooks_cli import run_hook
                run_hook(args.hook, args.harness)

        elif args.command == "instructions":
            from kai_mempalace.instructions_cli import run_instructions
            run_instructions(args.instr_command)

        elif args.command == "migrate":
            from kai_mempalace.migrate import migrate, rebuild_faiss, status as migrate_status
            base = str(Path(args.palace).expanduser().resolve())

            if args.status:
                s = migrate_status(base)
                print(f"Palace:   {s['path']}")
                print(f"Version:  {s['version']} (latest: {s['latest_version']})")
                print(f"Up to date: {s['up_to_date']}")
                print(f"Drawers:  {s.get('drawers', '?')}")
                print(f"Wings:    {s.get('wings', '?')}")
                print(f"Rooms:    {s.get('rooms', '?')}")
                print(f"Vectors:  {s.get('vectors', '?')}")
            elif args.rebuild_faiss:
                result = rebuild_faiss(base)
                print(f"FAISS rebuild: {result['vectors_rebuilt']} vectors")
            else:
                result = migrate(base, dry_run=args.dry_run)
                if args.dry_run:
                    print(f"Pending migrations: {result['migrations_applied'] or 'none'}")
                else:
                    print(f"Version: {result['version_before']} -> {result['version_after']}")
                    for m in result['migrations_applied']:
                        print(f"  Applied: {m}")

        elif args.command == "repair":
            from kai_mempalace.repair_utils import (
                confirm_destructive_action, rebuild_fts5, run_vacuum,
                sqlite_integrity_errors,
            )
            db_path = str(Path(args.palace).expanduser() / "data" / "metadata.db")
            if args.integrity:
                errors = sqlite_integrity_errors(db_path)
                if errors:
                    print(f"Integrity errors ({len(errors)}):")
                    for e in errors[:10]:
                        print(f"  - {e}")
                else:
                    print("Integrity check passed.")
            if args.vacuum:
                if confirm_destructive_action("VACUUM", db_path, assume_yes=args.yes):
                    run_vacuum(db_path)
                    print("VACUUM complete.")
            if args.rebuild_fts:
                if confirm_destructive_action("FTS5 rebuild", db_path, assume_yes=args.yes):
                    rebuild_fts5(db_path)
                    print("FTS5 index rebuilt.")
            if not any([args.integrity, args.vacuum, args.rebuild_fts]):
                errors = sqlite_integrity_errors(db_path)
                if errors:
                    print(f"Repair needed: {len(errors)} integrity issues")
                    for e in errors[:5]:
                        print(f"  - {e}")
                else:
                    print("No repair needed — palace is healthy.")

        elif args.command == "repair-status":
            from kai_mempalace.repair_utils import sqlite_drawer_count, sqlite_integrity_errors
            db_path = str(Path(args.palace).expanduser() / "data" / "metadata.db")
            count = sqlite_drawer_count(db_path)
            errors = sqlite_integrity_errors(db_path)
            print(f"Palace: {args.palace}")
            print(f"Drawers: {count or 0}")
            print(f"Integrity: {'PASS' if not errors else f'{len(errors)} issues'}")
            if errors:
                for e in errors[:5]:
                    print(f"  - {e}")

        elif args.command == "wake-up":
            from kai_mempalace.layers import MemoryStack
            stack = MemoryStack(palace)
            all_layers = stack.read_all(args.agent, last_n=args.last_n)
            for layer_num in sorted(all_layers):
                layer_name = MemoryStack.LAYER_NAMES[layer_num]
                entries = all_layers[layer_num]
                print(f"\n--- {layer_name} ({len(entries)} entries) ---")
                for e in entries:
                    print(f"  [{e['created_at']}] {e['content'][:200]}")

        elif args.command == "mcp":
            from kai_mempalace.mcp_server import run_server
            run_server(palace, host=args.host, port=args.port, transport=args.transport)

    finally:
        palace.close()


if __name__ == "__main__":
    main()
