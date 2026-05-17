"""
Discord parser (official Discord data export, not DiscordChatExporter).

NOTE: Discord's export only contains YOUR messages, not other people's.
Every message will be is_me=True. Useful for voice learning but no
conversational context.

Folder lookups are case-insensitive so renaming Messages -> messages
isn't necessary.

Usage:
    python scripts/parse_discord.py \\
        --package-root data/raw/discord \\
        --output       data/parsed/discord.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from schema import Message, to_jsonl


def find_subfolder(parent: Path, name: str) -> Optional[Path]:
    if not parent.exists():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def find_file(parent: Path, name: str) -> Optional[Path]:
    if not parent.exists():
        return None
    target = name.lower()
    for child in parent.iterdir():
        if child.is_file() and child.name.lower() == target:
            return child
    return None


def find_user_id(package_root: Path) -> str:
    account = find_subfolder(package_root, "account")
    if not account:
        return ""
    user_json = find_file(account, "user.json")
    if not user_json:
        return ""
    try:
        with user_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("id", ""))
    except (json.JSONDecodeError, OSError):
        return ""


def parse_messages_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        yield row


def parse_messages_csv(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "ID": row.get("ID", ""),
                "Timestamp": row.get("Timestamp", ""),
                "Contents": row.get("Contents", ""),
                "Attachments": row.get("Attachments", ""),
            }


def parse_timestamp(ts: str) -> str:
    ts = ts.strip()
    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat().replace("+00:00", "Z")
    except ValueError:
        return ts


def load_index(messages_root: Path) -> dict:
    idx = find_file(messages_root, "index.json")
    if not idx:
        return {}
    try:
        with idx.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def channel_dirs(messages_root: Path) -> list[Path]:
    if not messages_root.exists():
        return []
    return sorted(p for p in messages_root.iterdir() if p.is_dir() and p.name.lower().startswith("c"))


def parse_channel(channel_dir: Path, label: str) -> Iterable[Message]:
    channel_id = channel_dir.name.lstrip("cC")

    mj = find_file(channel_dir, "messages.json")
    mc = find_file(channel_dir, "messages.csv")
    if mj:
        rows = parse_messages_json(mj)
    elif mc:
        rows = parse_messages_csv(mc)
    else:
        return

    cj = find_file(channel_dir, "channel.json")
    is_group = False
    channel_type = "unknown"
    if cj:
        try:
            with cj.open("r", encoding="utf-8") as f:
                ch = json.load(f)
            channel_type = str(ch.get("type", "unknown")).lower()
            is_group = "group" in channel_type or channel_type == "guild_text"
        except (json.JSONDecodeError, OSError):
            pass

    conv_id = f"discord_channel_{channel_id}"

    for row in rows:
        text = row.get("Contents", "") or row.get("content", "")
        if not text or not text.strip():
            continue

        msg_id = str(row.get("ID", "") or row.get("id", ""))
        ts_raw = row.get("Timestamp", "") or row.get("timestamp", "")
        timestamp = parse_timestamp(ts_raw)

        yield Message(
            id=f"discord:{msg_id}" if msg_id else f"discord:{hash((conv_id, ts_raw, text)) & 0xFFFFFFFF:x}",
            platform="discord",
            timestamp=timestamp,
            conversation_id=conv_id,
            is_group=is_group,
            sender="me",
            text=text,
            is_me=True,
            reply_to_id=None,
            sender_real=None,
            meta={
                "channel_label": label,
                "channel_type": channel_type,
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse a Discord official data export.")
    ap.add_argument(
        "--package-root",
        type=Path,
        required=True,
        help="path to the unzipped Discord export (dir containing messages/)",
    )
    ap.add_argument("--output", type=Path, required=True, help="path to output JSONL")
    ap.add_argument(
        "--limit-channels",
        type=int,
        default=0,
        help="for testing: only parse this many channels (0 = all)",
    )
    args = ap.parse_args()

    messages_root = find_subfolder(args.package_root, "messages")
    if not messages_root:
        raise SystemExit(
            f"No messages/ folder under {args.package_root} (checked case-insensitively). Wrong path?"
        )

    user_id = find_user_id(args.package_root)
    if user_id:
        print(f"Detected Discord user_id: {user_id}")

    index = load_index(messages_root)
    dirs = channel_dirs(messages_root)
    if args.limit_channels:
        dirs = dirs[: args.limit_channels]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    channels_with_messages = 0

    with args.output.open("w", encoding="utf-8") as out:
        for cd in dirs:
            channel_id = cd.name.lstrip("cC")
            label = index.get(channel_id, cd.name)
            count_before = total
            for msg in parse_channel(cd, label):
                out.write(to_jsonl(msg) + "\n")
                total += 1
            if total > count_before:
                channels_with_messages += 1

    print(
        f"Parsed {channels_with_messages} channels with messages, "
        f"{total} total messages -> {args.output}"
    )


if __name__ == "__main__":
    main()
