"""
Stage 2: Cleaning.

Reads data/parsed/{platform}.jsonl, writes data/cleaned/{platform}.jsonl.

Operations:
  - URLs -> <url>
  - Media/attachment placeholders -> <media>
  - NFC unicode normalization
  - Whitespace normalization
  - Drop messages that become empty after cleaning

Preserves all messages (is_me True and False) so chunking has context.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path

# Local import: schema.Message
sys.path.insert(0, str(Path(__file__).parent))
from schema import Message  # type: ignore

PARSED_DIR = Path("data/parsed")
CLEANED_DIR = Path("data/cleaned")
PLATFORMS = ("instagram", "discord", "messenger")

# --- patterns ---------------------------------------------------------------

URL_RE = re.compile(
    r"""(?ix)
    \b
    (?:https?://|www\.)        # scheme or www
    [^\s<>"']+                 # body
    """,
)

# Platform-specific attachment / system placeholders we want to collapse.
# Match conservatively; if a real message happens to contain these as text,
# they're still semantically media references.
MEDIA_PATTERNS = [
    # Messenger
    re.compile(r"\[Photo\]", re.I),
    re.compile(r"\[Video\]", re.I),
    re.compile(r"\[Sticker\]", re.I),
    re.compile(r"\[GIF\]", re.I),
    re.compile(r"\[Audio\]", re.I),
    re.compile(r"\[Attachment\]", re.I),
    re.compile(r"\[File\]", re.I),
    # Instagram exported strings
    re.compile(r"Sent an attachment\.?", re.I),
    re.compile(r"Liked a message", re.I),
    re.compile(r"Reacted .{1,4} to your message", re.I),
    re.compile(r"Shared a story", re.I),
    re.compile(r"Sent a photo", re.I),
    re.compile(r"Sent a video", re.I),
    re.compile(r"Sent a voice message", re.I),
]

WHITESPACE_RE = re.compile(r"[ \t]+")
NEWLINES_RE = re.compile(r"\n{3,}")


# --- cleaning ---------------------------------------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""

    # 1. Unicode NFC (consistent codepoints for accented chars, emoji sequences)
    text = unicodedata.normalize("NFC", text)

    # 2. URLs -> token
    text = URL_RE.sub("<url>", text)

    # 3. Media placeholders -> token
    for pat in MEDIA_PATTERNS:
        text = pat.sub("<media>", text)

    # 4. Whitespace: collapse runs of spaces/tabs, cap blank lines at 2
    text = WHITESPACE_RE.sub(" ", text)
    text = NEWLINES_RE.sub("\n\n", text)

    return text.strip()


def clean_record(record: dict) -> dict | None:
    """Return cleaned record dict, or None if it should be dropped."""
    text = record.get("text", "") or ""
    cleaned = clean_text(text)
    if not cleaned:
        return None
    record["text"] = cleaned
    return record


# --- io ---------------------------------------------------------------------

def process_platform(platform: str) -> tuple[int, int]:
    src = PARSED_DIR / f"{platform}.jsonl"
    dst = CLEANED_DIR / f"{platform}.jsonl"
    if not src.exists():
        print(f"[skip] {src} not found")
        return 0, 0

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            cleaned = clean_record(record)
            if cleaned is None:
                dropped += 1
                continue
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped


def main() -> None:
    total_kept = total_dropped = 0
    for platform in PLATFORMS:
        kept, dropped = process_platform(platform)
        print(f"{platform:10s} kept={kept:>7d}  dropped={dropped:>6d}")
        total_kept += kept
        total_dropped += dropped
    print(f"{'TOTAL':10s} kept={total_kept:>7d}  dropped={total_dropped:>6d}")


if __name__ == "__main__":
    main()
