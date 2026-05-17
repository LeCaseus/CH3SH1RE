"""
Unified message schema.

Every parser (WhatsApp, Discord, Messenger, etc.) outputs JSONL
where each line conforms to Message. This is the contract that
keeps the pipeline sane.

Run this file directly to sanity-check by emitting an example.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import json


@dataclass
class Message:
    # stable unique id, namespaced by platform so collisions are impossible.
    # format: "{platform}:{platform_native_id}"
    id: str

    # one of: "whatsapp", "discord", "messenger", "instagram", "reddit", "x"
    platform: str

    # ISO 8601, always UTC. parsers must convert.
    timestamp: str

    # groups messages that belong to the same thread.
    # for DMs: stable identifier for the other party (anonymized)
    # for group chats: stable identifier for the group/channel
    conversation_id: str

    # whether the conversation has more than 2 participants
    is_group: bool

    # "me" if it's you, otherwise an anonymized handle like "person_a"
    # never store real names here. real names go in sender_real (optional, local only)
    sender: str

    # the actual text content
    text: str

    # THE CRITICAL FIELD. drives all downstream filtering.
    is_me: bool

    # id of the message this is a reply to, if any. uses the same id format.
    reply_to_id: Optional[str] = None

    # optional: real name kept locally for your own debugging.
    # strip before sharing the dataset anywhere.
    sender_real: Optional[str] = None

    # platform-specific extras parsers want to preserve.
    # don't depend on these downstream; they're for debugging.
    meta: dict = field(default_factory=dict)


def to_jsonl(msg: Message) -> str:
    """Serialize a Message to one JSONL line."""
    return json.dumps(asdict(msg), ensure_ascii=False)


def from_jsonl(line: str) -> Message:
    """Parse one JSONL line back to a Message."""
    data = json.loads(line)
    return Message(**data)


if __name__ == "__main__":
    # sanity check: emit an example and round-trip it
    example = Message(
        id="whatsapp:abc123",
        platform="whatsapp",
        timestamp="2024-03-15T14:23:11Z",
        conversation_id="person_a",
        is_group=False,
        sender="me",
        text="lol yeah that tracks",
        is_me=True,
        reply_to_id=None,
        sender_real="Buns",
    )
    line = to_jsonl(example)
    print("Serialized:", line)
    parsed = from_jsonl(line)
    print("Round-trip OK:", parsed == example)
