"""
WhatsApp parser.

WhatsApp exports look like this (one .txt file per chat, no media):

    [3/15/24, 2:23:11 PM] Buns: lol yeah that tracks
    [3/15/24, 2:23:45 PM] Alex: i told you
    [3/15/24, 2:24:02 PM] Buns: shut up alex
    [3/15/24, 2:24:30 PM] Alex: <Media omitted>

Some quirks WhatsApp throws at you:
  - "<Media omitted>" placeholder for stripped media
  - Multi-line messages continue on the next line WITHOUT a timestamp prefix
  - System messages: "Messages and calls are end-to-end encrypted..."
  - Format varies slightly by locale (date order, AM/PM vs 24h)
  - Apple devices use weird U+202F (narrow no-break space) before AM/PM

Usage:
    python scripts/parse_whatsapp.py \\
        --input  data/raw/whatsapp/chat_with_alex.txt \\
        --output data/parsed/whatsapp_alex.jsonl \\
        --me     "Buns" \\
        --conversation-id person_a
"""
from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from schema import Message, to_jsonl


# matches "[3/15/24, 2:23:11 PM] Sender: text" and variants.
# the [\u202f\s] handles the weird Apple narrow-no-break-space before AM/PM.
LINE_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:[\u202f\s][AP]M)?)\]\s+"
    r"(?P<sender>[^:]+?):\s+"
    r"(?P<text>.*)$"
)

# WhatsApp's "this chat is encrypted" and similar.
# add patterns here as you find them in your own exports.
SYSTEM_PATTERNS = [
    re.compile(r"end-to-end encrypted", re.IGNORECASE),
    re.compile(r"changed the group", re.IGNORECASE),
    re.compile(r"added .* to the group", re.IGNORECASE),
    re.compile(r"left$", re.IGNORECASE),
]

MEDIA_PLACEHOLDER = "<Media omitted>"


def parse_timestamp(date: str, time: str) -> str:
    """Parse WhatsApp date+time into ISO 8601 UTC string."""
    # normalize the weird narrow-no-break space
    time = time.replace("\u202f", " ").strip()
    combined = f"{date} {time}"

    # try a few formats. add more as you discover them in your exports.
    formats = [
        "%m/%d/%y %I:%M:%S %p",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(combined, fmt)
            # WhatsApp doesn't include timezone; we assume local and tag as naive.
            # if you care about exact UTC, you'd need to know the timezone you were in.
            return dt.isoformat() + "Z"
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp: {combined!r}")


def is_system_message(text: str) -> bool:
    return any(p.search(text) for p in SYSTEM_PATTERNS)


def make_id(platform: str, conversation_id: str, timestamp: str, sender: str, text: str) -> str:
    """WhatsApp has no native message ids, so we hash content for stability."""
    h = hashlib.sha1(
        f"{conversation_id}|{timestamp}|{sender}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{platform}:{h}"


def parse_file(
    path: Path,
    me_real_name: str,
    conversation_id: str,
    is_group: bool = False,
) -> list[Message]:
    """Parse a WhatsApp export into a list of Message objects."""
    messages: list[Message] = []
    current: Optional[dict] = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            m = LINE_RE.match(line)

            if m:
                # flush the previous message before starting a new one
                if current is not None:
                    messages.append(_finalize(current, me_real_name, conversation_id, is_group))

                current = {
                    "date": m.group("date"),
                    "time": m.group("time"),
                    "sender": m.group("sender").strip(),
                    "text_lines": [m.group("text")],
                }
            else:
                # continuation line of the previous message
                if current is not None:
                    current["text_lines"].append(line)
                # else: junk before the first parseable line; ignore.

    # don't forget the last one
    if current is not None:
        messages.append(_finalize(current, me_real_name, conversation_id, is_group))

    return messages


def _finalize(
    current: dict,
    me_real_name: str,
    conversation_id: str,
    is_group: bool,
) -> Optional[Message]:
    text = "\n".join(current["text_lines"]).strip()
    sender_real = current["sender"]

    # skip system messages and pure media-omitted lines.
    # we return them as None and filter below... actually let's just skip here.
    # but a return type of Optional[Message] would force callers to filter.
    # easier: return a Message tagged as system in meta, filter in cleaning stage.
    # for now: drop system messages outright. media placeholders we keep as empty text
    # so the cleaner can decide.

    is_me = (sender_real == me_real_name)
    sender = "me" if is_me else _anonymize(sender_real)

    timestamp = parse_timestamp(current["date"], current["time"])

    msg = Message(
        id=make_id("whatsapp", conversation_id, timestamp, sender, text),
        platform="whatsapp",
        timestamp=timestamp,
        conversation_id=conversation_id,
        is_group=is_group,
        sender=sender,
        text=text,
        is_me=is_me,
        reply_to_id=None,  # WhatsApp text export doesn't preserve reply threading
        sender_real=sender_real,
        meta={
            "is_system": is_system_message(text),
            "is_media_placeholder": text == MEDIA_PLACEHOLDER,
        },
    )
    return msg


def _anonymize(real_name: str) -> str:
    """Stable anonymized handle for a non-me sender within a conversation."""
    # simple approach: hash the real name. same name -> same handle across files.
    # if you'd rather assign person_a / person_b / person_c per-conversation,
    # that's a job for the cleaning stage where you can see the full pool.
    h = hashlib.sha1(real_name.encode("utf-8")).hexdigest()[:8]
    return f"user_{h}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse a WhatsApp export.")
    ap.add_argument("--input", type=Path, required=True, help="path to WhatsApp .txt export")
    ap.add_argument("--output", type=Path, required=True, help="path to output JSONL")
    ap.add_argument("--me", type=str, required=True, help="your display name as it appears in the chat")
    ap.add_argument("--conversation-id", type=str, required=True, help="stable id for this conversation")
    ap.add_argument("--group", action="store_true", help="set if this is a group chat")
    args = ap.parse_args()

    messages = parse_file(args.input, args.me, args.conversation_id, args.group)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(to_jsonl(msg) + "\n")

    me_count = sum(1 for m in messages if m.is_me)
    print(f"Parsed {len(messages)} messages ({me_count} from you) -> {args.output}")


if __name__ == "__main__":
    main()
