"""
Stage 4: Formatting.

Reads data/chunked/{platform}.jsonl, writes data/formatted/train.jsonl.

Strategy:
  - Drop Discord entirely (no other-side context = unusable for conversational)
  - For each chunk, build a list of turns:
      * sender == "me"  -> role="assistant"
      * sender != "me"  -> role="user"
  - Merge consecutive same-role messages into one turn (joined with "\n")
  - Drop chunks with zero assistant turns

Output: one JSON object per line, model-agnostic intermediate format:
  {
    "chunk_id": "...",
    "platform": "...",
    "conversation_id": "...",
    "messages": [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
  }

Apply a tokenizer's chat template (e.g. Qwen, Llama-3) at training time:
  text = tokenizer.apply_chat_template(record["messages"], tokenize=False)
"""

from __future__ import annotations

import json
from pathlib import Path

CHUNKED_DIR = Path("data/chunked")
FORMATTED_DIR = Path("data/formatted")
PLATFORMS = ("instagram", "messenger")  # Discord excluded: no other-side context

OUTPUT_PATH = FORMATTED_DIR / "train.jsonl"


def role_for(msg: dict) -> str:
    return "assistant" if msg.get("is_me") else "user"


def chunk_to_turns(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages into single turns."""
    turns: list[dict] = []
    for m in messages:
        role = role_for(m)
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + text
        else:
            turns.append({"role": role, "content": text})
    return turns


def has_assistant_turn(turns: list[dict]) -> bool:
    return any(t["role"] == "assistant" for t in turns)


def process_platform(platform: str, fout) -> dict[str, int]:
    src = CHUNKED_DIR / f"{platform}.jsonl"
    if not src.exists():
        print(f"[skip] {src} not found")
        return {"in": 0, "out": 0, "dropped_no_me": 0}

    stats = {"in": 0, "out": 0, "dropped_no_me": 0}
    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["in"] += 1

            turns = chunk_to_turns(chunk.get("messages", []))
            if not has_assistant_turn(turns):
                stats["dropped_no_me"] += 1
                continue

            record = {
                "chunk_id": chunk["chunk_id"],
                "platform": chunk["platform"],
                "conversation_id": chunk["conversation_id"],
                "messages": turns,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["out"] += 1
    return stats


def main() -> None:
    FORMATTED_DIR.mkdir(parents=True, exist_ok=True)
    totals = {"in": 0, "out": 0, "dropped_no_me": 0}
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for platform in PLATFORMS:
            s = process_platform(platform, fout)
            print(
                f"{platform:10s} "
                f"in={s['in']:>6d}  out={s['out']:>6d}  "
                f"dropped_no_me={s['dropped_no_me']:>5d}"
            )
            for k in totals:
                totals[k] += s[k]
    print(
        f"{'TOTAL':10s} "
        f"in={totals['in']:>6d}  out={totals['out']:>6d}  "
        f"dropped_no_me={totals['dropped_no_me']:>5d}"
    )
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
