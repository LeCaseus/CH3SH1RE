"""
Instagram DM parser.

Meta's export ships Instagram DMs as JSON files at:
    <export_root>/your_instagram_activity/messages/inbox/<thread_name>/message_1.json
    (and message_2.json, message_3.json, ... for long threads)

THE BIG GOTCHA: Meta double-encodes UTF-8 as Latin-1. Emoji and accented
characters arrive as 'ð\u009f\u0098\u0082' instead of '😂'. We fix this
by re-interpreting strings as latin-1 bytes then decoding as utf-8.

Usage:
    python scripts/parse_instagram.py \\
        --inbox-root data/raw/instagram/your_instagram_activity/messages/inbox \\
        --output     data/parsed/instagram.jsonl \\
        --me         "Buns" "buns_real" "bunsythebun"
"""
from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from schema import Message, to_jsonl


def fix_mojibake(s: str) -> str:
    """Undo Meta's latin-1/utf-8 double-encoding."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def fix_in_place(obj):
    if isinstance(obj, str):
        return fix_mojibake(obj)
    if isinstance(obj, list):
        return [fix_in_place(x) for x in obj]
    if isinstance(obj, dict):
        return {k: fix_in_place(v) for k, v in obj.items()}
    return obj


def anonymize(real_name: str) -> str:
    h = hashlib.sha1(real_name.encode("utf-8")).hexdigest()[:8]
    return f"user_{h}"


def thread_conversation_id(
    thread_dir: Path, participants: list[str], me_aliases: set[str]
) -> str:
    others = [p for p in participants if p not in me_aliases and p.strip()]
    if len(others) == 1:
        return anonymize(others[0])
    # Group chats, or 1:1s where the other participant has an empty name
    # (deactivated/deleted accounts). Hash the thread folder for uniqueness.
    h = hashlib.sha1(thread_dir.name.encode("utf-8")).hexdigest()[:8]
    prefix = "group" if len(participants) > 2 else "user"
    return f"{prefix}_{h}"


def make_message_id(timestamp_ms: int, sender: str, text: str) -> str:
    h = hashlib.sha1(
        f"{timestamp_ms}|{sender}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"messenger:{h}"


def parse_thread(thread_dir: Path, me_aliases: set[str]) -> Iterable[Message]:
    json_files = sorted(thread_dir.glob("message_*.json"))
    if not json_files:
        return

    with json_files[0].open("r", encoding="utf-8") as f:
        first = fix_in_place(json.load(f))

    participants = [p["name"] for p in first.get("participants", [])]
    is_group = len(participants) > 2
    conv_id = thread_conversation_id(thread_dir, participants, me_aliases)

    for jf in json_files:
        with jf.open("r", encoding="utf-8") as f:
            data = fix_in_place(json.load(f))

        for raw in data.get("messages", []):
            text = raw.get("content", "")
            if not text:
                continue

            sender_real = raw.get("sender_name", "unknown")
            is_me = (sender_real in me_aliases)
            sender = "me" if is_me else anonymize(sender_real)

            ts_ms = raw.get("timestamp_ms", 0)
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            timestamp = dt.isoformat().replace("+00:00", "Z")

            yield Message(
                id=make_message_id(ts_ms, sender, text),
                platform="messenger",
                timestamp=timestamp,
                conversation_id=conv_id,
                is_group=is_group,
                sender=sender,
                text=text,
                is_me=is_me,
                reply_to_id=None,
                sender_real=sender_real,
                meta={
                    "thread_folder": thread_dir.name,
                    "is_unsent": raw.get("is_unsent", False),
                    "has_reactions": bool(raw.get("reactions")),
                },
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Instagram DM exports from Meta.")
    ap.add_argument(
        "--inbox-root",
        type=Path,
        required=True,
        help="path to .../messages/inbox/ directory",
    )
    ap.add_argument("--output", type=Path, required=True, help="path to output JSONL")
    ap.add_argument(
        "--me",
        type=str,
        nargs="+",
        required=True,
        help="your name(s) as they appear in sender_name. pass multiple if you changed display names.",
    )
    ap.add_argument(
        "--limit-threads",
        type=int,
        default=0,
        help="for testing: only parse this many threads (0 = all)",
    )
    args = ap.parse_args()

    me_aliases = set(args.me)
    print(f"Treating these names as you: {sorted(me_aliases)}")

    if not args.inbox_root.exists():
        raise SystemExit(f"inbox-root does not exist: {args.inbox_root}")

    thread_dirs = sorted(p for p in args.inbox_root.iterdir() if p.is_dir())
    if args.limit_threads:
        thread_dirs = thread_dirs[: args.limit_threads]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    me_count = 0
    threads_processed = 0

    with args.output.open("w", encoding="utf-8") as out:
        for td in thread_dirs:
            count_before = total
            for msg in parse_thread(td, me_aliases):
                out.write(to_jsonl(msg) + "\n")
                total += 1
                if msg.is_me:
                    me_count += 1
            if total > count_before:
                threads_processed += 1

    print(
        f"Parsed {threads_processed} threads, {total} messages "
        f"({me_count} from you) -> {args.output}"
    )


if __name__ == "__main__":
    main()
