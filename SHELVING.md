# SHELVING.md

Hey future me. Here's how to park this project and resume cleanly on a new machine. Read this top to bottom.

## Before you shelve it

You need to back up two things, separately. The repo is on GitHub but it only has the code. The data and the trained model are gitignored (as they're private and big).

### What to archive

```
ch3sh1re-shelved-<date>/
└── data/
    └── raw/        # the social media exports. irreplaceable.
```

That's it. Everything else is either on GitHub (the code) or regenerable from `data/raw/` (the parsed/cleaned/chunked/formatted intermediates, and eventually the trained model).

You may or may not want to keep the old trained adapter (`outputs/qwen25-3b-ch3sh1re/final/`, ~100 MB). It's not great — the model needs retraining anyway. But it's small, so keep it if you want a baseline to compare future runs against. Optional.

**Do not bother archiving:**
- `data/parsed/`, `data/cleaned/`, `data/chunked/`, `data/formatted/` — regenerable in minutes by running the scripts
- `outputs/qwen25-3b-ch3sh1re/checkpoint-*` — three mid-training snapshots, hundreds of MB each. Useless because you're changing the data and config for the retrain.
- `outputs/qwen25-3b-ch3sh1re/merged-fp16/` — ~6 GB. One minute to regenerate from the adapter.
- `outputs/qwen25-3b-ch3sh1re/*.gguf` — ~8 GB combined. Regenerable.
- `unsloth_compiled_cache/`, `.venv/`, `__pycache__/` — all auto-generated.

### Sanity check the sizes before tarring

```fish
du -sh data/raw
```

Make sure it's there and the size matches what you expect (tens of GB — Instagram media dump is the bulk). If it's tiny or missing, find out *now*, not in six months.

---

## When you come back: resume on a new machine

Assuming a fresh Linux install with Python available and a GPU (hopefully bigger than 4 GB this time).

### 1. Clone the repo

```fish
git clone https://github.com/LeCaseus/CH3SH1RE.git
cd CH3SH1RE
```

### 2. Restore the raw data

Copy `data/raw/` from your archive into the repo

Verify:
```fish
du -sh data/raw
ls data/raw
```
Should see `messenger/`, `instagram/`, `discord/` directories.

### 3. Set up Python environment

```fish
python3.11 -m venv .venv
source .venv/bin/activate.fish
pip install uv
uv pip install unsloth --torch-backend=auto
```

If the new GPU is still small (≤8 GB), add to `~/.config/fish/config.fish`:
```fish
set -gx PYTORCH_ALLOC_CONF expandable_segments:True
```

### 4. Fix the known issues before retraining

Open `scripts/clean.py` and add Messenger system patterns to `MEDIA_PATTERNS`. These are the leaks that polluted the last training run:
```python
re.compile(r"You can now call each other", re.I),
re.compile(r"You changed the theme to", re.I),
re.compile(r"You set the quick reaction to", re.I),
re.compile(r".+ sent an attachment\.", re.I),
```
(Add more as you find them in the data — check `data/cleaned/messenger.jsonl` after running clean.py for anything that looks like Facebook system noise.)

Open `scripts/train.py` and consider:
- Removing `train_on_responses_only` (training on full conversation may improve context-following)
- Setting a distinct pad token (not equal to EOS)
- Bumping `MAX_SEQ_LEN` to 2048 if VRAM allows
- Bumping epochs to 4-6
- Going back to Qwen2.5-7B if VRAM allows

Read the README's "What went wrong & priorities for retraining" section first.

### 5. Run Phase 1 (parsing → formatted dataset)

The parsers need flags — your name(s) as they appear in the exports, and the paths to the inbox dirs. Adjust accordingly:

```fish
python scripts/parse_messenger.py \
    --inbox-root data/raw/messenger/your_facebook_activity/messages/inbox \
    --output data/parsed/messenger.jsonl \
    --me "<your messenger display name(s)>"

python scripts/parse_instagram.py \
    --inbox-root data/raw/instagram/your_instagram_activity/messages/inbox \
    --output data/parsed/instagram.jsonl \
    --me "<your instagram display name(s)>"

# Discord parser exists but its output is dropped at format stage — skip unless you have a one-sided-context strategy

python scripts/clean.py
python scripts/chunk.py
python scripts/format.py
python scripts/split.py
```

After this you should have `data/formatted/train_split.jsonl` and `data/formatted/val.jsonl`. Check counts look reasonable (should be tens of thousands of examples, ~95/5 split).

### 6. Run Phase 2 (training)

```fish
python scripts/train.py
```

This takes hours. On the old hardware it was 8h 14m for 2 epochs. New GPU should be faster. Watch eval loss — last run plateaued at 3.44 which was too high. Aim lower.

Output goes to `outputs/qwen25-3b-ch3sh1re/final/` (or whatever you renamed it to).

### 7. Run Phase 3 (inference)

```fish
python scripts/merge.py
```

Then build llama.cpp (CPU is fine, GPU is a rabbit hole on Fedora — see README):
```fish
cd ~/BuiltFromSource
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
python3.11 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements/requirements-convert_hf_to_gguf.txt
cmake -B build
cmake --build build --config Release -j4
```

Convert, quantize, run:
```fish
python convert_hf_to_gguf.py \
    /path/to/CH3SH1RE/outputs/qwen25-3b-ch3sh1re/merged-fp16 \
    --outfile /path/to/CH3SH1RE/outputs/qwen25-3b-ch3sh1re/ch3sh1re-fp16.gguf \
    --outtype f16

./build/bin/llama-quantize \
    /path/to/CH3SH1RE/outputs/qwen25-3b-ch3sh1re/ch3sh1re-fp16.gguf \
    /path/to/CH3SH1RE/outputs/qwen25-3b-ch3sh1re/ch3sh1re-q4_k_m.gguf \
    Q4_K_M

./build/bin/llama-cli \
    -m /path/to/CH3SH1RE/outputs/qwen25-3b-ch3sh1re/ch3sh1re-q4_k_m.gguf \
    --chat-template chatml \
    -sys "<system prompt>" \
    --temp 0.6 --top-p 0.85 --top-k 30 --repeat-penalty 1.2 \
    -n 100 -cnv
```

## If anything feels off

Read the README. It has the full context on what I tried, what worked, what didn't, and why the last model output was bad.
