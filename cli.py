#!/usr/bin/env python3
"""
Kai MemPalace CLI — manage your memory palace from the command line.

Usage:
    kai-mempalace init                         # Create a new palace
    kai-mempalace status                       # Show palace status

    kai-mempalace add --wing <w> --room <r> --content "..." [--meta '{"k":"v"}']
    kai-mempalace search <query> [--wing w] [--room r] [--limit 10]
    kai-mempalace get <drawer-id>
    kai-mempalace list [--wing w] [--room r] [--limit 20]

    kai-mempalace wings                        # List wings
    kai-mempalace rooms [--wing w]             # List rooms

    kai-mempalace kg-add --subject s --predicate p --object o [--source src]
    kai-mempalace kg-query <entity> [--as-of DATE]
    kai-mempalace kg-invalidate --subject s --predicate p --object o

    kai-mempalace diary --agent kai --entry "..." [--topic general]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palace import Palace

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Kai MemPalace CLI")
    parser.add_argument("--palace", default="~/.kai-palace",
                        help="Path to the palace directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # init
    sub.add_parser("init", help="Initialize a new palace")

    # status
    sub.add_parser("status", help="Show palace status")

    # add
    p_add = sub.add_parser("add", help="Add a drawer")
    p_add.add_argument("--wing", "-w", required=True)
    p_add.add_argument("--room", "-r", required=True)
    p_add.add_argument("--content", "-c", required=True)
    p_add.add_argument("--meta", "-m", default="{}")
    p_add.add_argument("--source", "-s", default="")

    # search
    p_srch = sub.add_parser("search", help="Search the palace")
    p_srch.add_argument("query", nargs="?", default="")
    p_srch.add_argument("--wing", "-w")
    p_srch.add_argument("--room", "-r")
    p_srch.add_argument("--limit", "-l", type=int, default=10)

    # get
    p_get = sub.add_parser("get", help="Get a drawer by ID")
    p_get.add_argument("drawer_id")

    # list
    p_list = sub.add_parser("list", help="List drawers")
    p_list.add_argument("--wing", "-w")
    p_list.add_argument("--room", "-r")
    p_list.add_argument("--limit", "-l", type=int, default=20)
    p_list.add_argument("--offset", "-o", type=int, default=0)

    # wings
    p_wing = sub.add_parser("wings", help="List wings")
    p_wing.add_argument("--verbose", "-v", action="store_true")

    # rooms
    p_room = sub.add_parser("rooms", help="List rooms")
    p_room.add_argument("--wing", "-w")

    # delete
    p_del = sub.add_parser("delete", help="Delete a drawer")
    p_del.add_argument("drawer_id")

    # knowledge graph
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
    p_kgq.add_argument("--direction", default="both", choices=["outgoing", "incoming", "both"])
    p_kgq.add_argument("--all", "-a", action="store_true")

    p_kgi = sub.add_parser("kg-invalidate", help="Invalidate KG fact")
    p_kgi.add_argument("--subject", "-s", required=True)
    p_kgi.add_argument("--predicate", "-p", required=True)
    p_kgi.add_argument("--object", "-o", required=True)
    p_kgi.add_argument("--ended")

    # diary
    p_diary = sub.add_parser("diary", help="Write a diary entry")
    p_diary.add_argument("--agent", "-a", required=True)
    p_diary.add_argument("--entry", "-e", required=True)
    p_diary.add_argument("--topic", "-t", default="general")
    p_diary.add_argument("--wing", "-w", default="")

    p_diaryr = sub.add_parser("diary-read", help="Read diary entries")
    p_diaryr.add_argument("--agent", "-a", required=True)
    p_diaryr.add_argument("--last-n", type=int, default=10)

    # dedup check
    p_dedup = sub.add_parser("check-dup", help="Check for duplicate content")
    p_dedup.add_argument("content")
    p_dedup.add_argument("--threshold", type=float, default=0.9)

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
                                     wing=args.wing, room=args.room)
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
                    desc = f" — {w['description']}" if w['description'] and args.verbose else ""
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
                                         as_of=args.as_of,
                                         direction=args.direction)
            if not facts:
                print("No facts found.")
            else:
                for f in facts:
                    valid = f" [{f['valid_from']} → {f['valid_to'] or 'now'}]" if f['valid_from'] else ""
                    print(f"  {f['subject']} -- {f['predicate']} -- {f['object']}{valid}")

        elif args.command == "kg-invalidate":
            n = palace.kg.invalidate(args.subject, args.predicate, args.object, args.ended)
            print(f"Invalidated {n} fact(s).")

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

    finally:
        palace.close()


if __name__ == "__main__":
    main()
