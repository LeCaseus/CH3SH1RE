"""
Stage 3: Chunking.

Reads data/cleaned/{platform}.jsonl, writes data/chunked/{platform}.jsonl.

Strategy:
  - Group messages by conversation_id, sort by timestamp
  - Split into sessions on >= 1h silence
  - Drop sessions with < 3 messages UNLESS at least one is_me=True
  - Split oversized sessions (> 2048 tokens) into multiple chunks at message
    boundaries, using the Qwen2.5 tokenizer for accurate counts

Output: one JSON object per line, where each object represents a chunk:
  {
    "chunk_id": "<platform>_<conversation_id>_<session_idx>_<chunk_idx>",
    "platform": "...",
    "conversation_id": "...",
    "messages": [ <Message dict>, ... ],
    "token_count": int,
    "start_ts": float,
    "end_ts": float,
  }
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer  # type: ignore

sys.path.insert(0, str(Path(__file__).parent))

CLEANED_DIR = Path("data/cleaned")
CHUNKED_DIR = Path("data/chunked")
PLATFORMS = ("instagram", "discord", "messenger")

SESSION_GAP_SECONDS = 60 * 60         # 1 hour
MAX_TOKENS_PER_CHUNK = 2048
MIN_MESSAGES_IF_NO_ME = 3             # sessions shorter than this need is_me=True

TOKENIZER_NAME = "Qwen/Qwen2.5-7B"

# --- tokenizer --------------------------------------------------------------

print(f"[init] loading tokenizer: {TOKENIZER_NAME}")
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def count_tokens(text: str) -> int:
    # Fast path; no special tokens, we just need a length estimate per message.
    return len(TOKENIZER.encode(text, add_special_tokens=False))


# --- core logic -------------------------------------------------------------

def parse_ts(ts) -> float:
    """Accept ISO 8601 string, int/float seconds, or int ms. Return unix seconds."""
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        # Heuristic: 13-digit ints are milliseconds
        return ts / 1000.0 if ts > 1e12 else float(ts)
    if isinstance(ts, str):
        # Handle trailing Z (UTC)
        s = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def load_messages(platform: str) -> list[dict]:
    src = CLEANED_DIR / f"{platform}.jsonl"
    if not src.exists():
        print(f"[skip] {src} not found")
        return []
    msgs: list[dict] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            m["timestamp"] = parse_ts(m.get("timestamp"))
            msgs.append(m)
    return msgs


def group_by_conversation(msgs: list[dict]) -> dict[str, list[dict]]:
    convos: dict[str, list[dict]] = defaultdict(list)
    for m in msgs:
        convos[m.get("conversation_id", "unknown")].append(m)
    for cid in convos:
        convos[cid].sort(key=lambda m: m.get("timestamp", 0))
    return convos


def split_sessions(messages: list[dict]) -> list[list[dict]]:
    """Split a conversation's messages into sessions on >=1h gaps."""
    if not messages:
        return []
    sessions: list[list[dict]] = []
    current: list[dict] = [messages[0]]
    for prev, curr in zip(messages, messages[1:]):
        gap = curr.get("timestamp", 0) - prev.get("timestamp", 0)
        if gap >= SESSION_GAP_SECONDS:
            sessions.append(current)
            current = [curr]
        else:
            current.append(curr)
    sessions.append(current)
    return sessions


def session_qualifies(session: list[dict]) -> bool:
    """Keep session if it has >=3 messages, or if any message is is_me=True."""
    if len(session) >= MIN_MESSAGES_IF_NO_ME:
        return True
    return any(m.get("is_me") for m in session)


def split_oversized(
    session: list[dict],
) -> list[tuple[list[dict], int]]:
    """
    Split a session into chunks of <= MAX_TOKENS_PER_CHUNK at message boundaries.
    Returns list of (chunk_messages, token_count).

    Pre-tokenizes each message once; greedy pack.
    """
    msg_tokens = [count_tokens(m.get("text", "") or "") for m in session]

    chunks: list[tuple[list[dict], int]] = []
    cur_msgs: list[dict] = []
    cur_tokens = 0

    for m, t in zip(session, msg_tokens):
        # If a single message is itself > MAX_TOKENS, it still gets its own chunk.
        if cur_tokens + t > MAX_TOKENS_PER_CHUNK and cur_msgs:
            chunks.append((cur_msgs, cur_tokens))
            cur_msgs = []
            cur_tokens = 0
        cur_msgs.append(m)
        cur_tokens += t

    if cur_msgs:
        chunks.append((cur_msgs, cur_tokens))
    return chunks


def build_chunk_record(
    platform: str,
    conversation_id: str,
    session_idx: int,
    chunk_idx: int,
    messages: list[dict],
    token_count: int,
) -> dict:
    return {
        "chunk_id": f"{platform}_{conversation_id}_{session_idx}_{chunk_idx}",
        "platform": platform,
        "conversation_id": conversation_id,
        "messages": messages,
        "token_count": token_count,
        "start_ts": messages[0].get("timestamp"),
        "end_ts": messages[-1].get("timestamp"),
    }


# --- io ---------------------------------------------------------------------

def process_platform(platform: str) -> dict[str, int]:
    msgs = load_messages(platform)
    if not msgs:
        return {"chunks": 0, "sessions": 0, "dropped_sessions": 0, "messages": 0}

    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    dst = CHUNKED_DIR / f"{platform}.jsonl"

    convos = group_by_conversation(msgs)
    stats = {"chunks": 0, "sessions": 0, "dropped_sessions": 0, "messages": 0}

    with dst.open("w", encoding="utf-8") as fout:
        for conversation_id, conv_msgs in convos.items():
            sessions = split_sessions(conv_msgs)
            for s_idx, session in enumerate(sessions):
                if not session_qualifies(session):
                    stats["dropped_sessions"] += 1
                    continue
                stats["sessions"] += 1
                chunks = split_oversized(session)
                for c_idx, (chunk_msgs, tok_count) in enumerate(chunks):
                    record = build_chunk_record(
                        platform, conversation_id, s_idx, c_idx,
                        chunk_msgs, tok_count,
                    )
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["chunks"] += 1
                    stats["messages"] += len(chunk_msgs)
    return stats


def main() -> None:
    totals = {"chunks": 0, "sessions": 0, "dropped_sessions": 0, "messages": 0}
    for platform in PLATFORMS:
        s = process_platform(platform)
        print(
            f"{platform:10s} "
            f"chunks={s['chunks']:>7d}  "
            f"sessions={s['sessions']:>7d}  "
            f"dropped={s['dropped_sessions']:>6d}  "
            f"msgs={s['messages']:>7d}"
        )
        for k in totals:
            totals[k] += s[k]
    print(
        f"{'TOTAL':10s} "
        f"chunks={totals['chunks']:>7d}  "
        f"sessions={totals['sessions']:>7d}  "
        f"dropped={totals['dropped_sessions']:>6d}  "
        f"msgs={totals['messages']:>7d}"
    )


if __name__ == "__main__":
    main()
