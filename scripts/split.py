"""
Split data/formatted/train.jsonl into train_split.jsonl and val.jsonl
by conversation_id (~5% val).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
VAL_FRAC = 0.05
SRC = Path("data/formatted/train.jsonl")
OUT_TRAIN = Path("data/formatted/train_split.jsonl")
OUT_VAL = Path("data/formatted/val.jsonl")


def main():
    random.seed(SEED)

    # Group chunks by conversation_id
    by_conv = defaultdict(list)
    with SRC.open() as f:
        for line in f:
            ex = json.loads(line)
            by_conv[ex["conversation_id"]].append(ex)

    conv_ids = list(by_conv.keys())
    random.shuffle(conv_ids)

    n_val = max(1, int(len(conv_ids) * VAL_FRAC))
    val_ids = set(conv_ids[:n_val])
    train_ids = set(conv_ids[n_val:])

    n_train_ex = n_val_ex = 0
    with OUT_TRAIN.open("w") as ft, OUT_VAL.open("w") as fv:
        for cid, chunks in by_conv.items():
            out = ft if cid in train_ids else fv
            for ex in chunks:
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                if cid in train_ids:
                    n_train_ex += 1
                else:
                    n_val_ex += 1

    print(f"Conversations: {len(train_ids)} train / {len(val_ids)} val")
    print(f"Examples:      {n_train_ex} train / {n_val_ex} val")
    print(f"Val fraction:  {n_val_ex / (n_train_ex + n_val_ex):.2%}")


if __name__ == "__main__":
    main()
