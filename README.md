# CH3SH1RE

> A local, privacy-first AI assistant with persistent memory, intent routing, and web research — no cloud, no accounts, no data leaving your machine.

---

## What is this?

CH3SH1RE is a personal AI assistant that I made to run entirely on my local hardware via [Ollama](https://ollama.com/). It was built out of a simple principle: I keep my conversations and my data to myself. Completely private, no servers, and no one to use my data to train their models.
Beyond privacy, it was also a learning project. A way to understand how LLM-backed applications actually work by building one from scratch.

> *[I'll add a screenshot of the UI next time :p]*

---

## What I built into it

- **Intent-aware routing** — Detects what I'm asking for (career stuff, writing, research, planning, study, etc.) and switches to a different system prompt tailored for that context.
- **Persistent memory** — Stores my conversation history in a local SQLite database. Pulls in recent exchanges and searches past conversations for relevant context when I ask something.
- **Structured fact extraction** — After each conversation, silently extracts key facts about me (preferences, location, goals) and stores them as key-value pairs. Those get injected into future responses so I don't have to repeat myself.
- **Web research** — One-shot search for time-sensitive stuff (news, prices, current events) and a multi-round loop for deeper questions. Uses DuckDuckGo so no API key needed.
- **File reading** — I can drop in a PDF, DOCX, or TXT and ask questions about it.
- **Streaming responses** — Words render as they come in rather than waiting for the full reply.
- **Chain-of-thought reasoning** — For heavier intents (career, research, study, planning), it runs a silent reasoning pass before giving me the actual answer.
- **Fact seeding** — I can pre-load facts about myself via a `seed_facts.json` so it already knows the basics from the first conversation.

---

## How I structured it

```
CH3SH1RE/
├── assistant/
│   ├── __init__.py
│   ├── extractor.py     # Pulls structured facts from each conversation
│   ├── llm.py           # Ollama interface — ask_llm, think, stream_llm_chunks
│   ├── main.py          # FastAPI app — routes, file handling, streaming endpoint
│   ├── memory.py        # SQLite memory — conversations + user_facts tables
│   ├── prompts.py       # All system prompts, one function per intent
│   ├── researcher.py    # Multi-round web research loop
│   ├── router.py        # Intent detection, search triggering, message assembly
│   └── tools.py         # Web search (DuckDuckGo) and file reading
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/                # SQLite DB lives here (gitignored)
├── uploads/             # Temporary file uploads (gitignored)
├── seed.py              # Seeds user_facts table from seed_facts.json
├── run.py               # Starts the uvicorn server
└── requirements.txt     # [Add this if you haven't yet]
```

---

## What I ran it on

### Hardware

| Component | What I used |
|-----------|-------------|
| CPU | AMD Ryzen 5 5500U |
| RAM | 16 GB |
| GPU | NVIDIA RTX 3050 4GB VRAM |

Ollama can run CPU-only — it'll just be slower without a GPU.

### Software

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Model pulled: `ollama pull qwen3:4b` *(or whatever you set in `llm.py`)*

---

## Running it

```bash
# 1. Clone the repo
git clone https://github.com/LeCaseus/CH3SH1RE.git
cd CH3SH1RE

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and the model is pulled
ollama pull qwen3:4b

# 5. (Optional) Seed it with facts about yourself
#    Edit seed_facts.json first, then:
python seed.py

# 6. Start the app
python run.py
```

Then open `http://127.0.0.1:8000` in a browser.

---

## Changing the model

In `assistant/llm.py`:

```python
MODEL = "qwen3:4b"   # swap for any model pulled in Ollama
```

Bigger models give better answers. I used quantized versions (Q4/Q5) to stay within my VRAM.

---

## Seeding facts about yourself

Create `seed_facts.json` in the project root:

```json
{
  "facts": {
    "name": "Your name",
    "location": "Your city",
    "occupation": "What you do"
  },
  "instructions": {
    "instruction_tone": "Be concise and direct",
    "instruction_format": "Avoid bullet points unless I ask"
  }
}
```

Run `python seed.py` and those get stored in the database, injected into relevant future conversations.

---

## How the memory works

Everything lives in a local SQLite database (`data/memory.db`) across two tables:

- **`conversations`** — raw exchange log with timestamps
- **`user_facts`** — structured key/value facts extracted from conversations

When I send a message, it pulls the 6 most recent exchanges plus up to 2 keyword-matched past ones and injects them into the system prompt. After each response, a silent pass extracts new facts and upserts them into `user_facts` for use in future relevant conversations.

---

## How intent routing works

The router matches keywords in my message against a list and picks the right system prompt:

| Intent | Triggered by | Prompt style |
|--------|-------------|--------------|
| `career` | "cover letter", "job application", "CV" | Professional career advisor |
| `writing` | "draft", "rewrite", "proofread" | Writer and editor |
| `research` | "compare", "pros and cons", "vs" | Research assistant |
| `study` | "flashcard", "quiz", "teach me" | Study assistant |
| `planning` | "plan", "schedule", "brainstorm" | Planning assistant |
| `personal` | "recipe", "relationship", "budget" | Personal assistant |
| `summarise` | "summarise", "tldr", "key points" | Summariser |
| `chat` | (everything else) | General conversation |

---

## Things I want to add eventually

- [ ] Vector embeddings for proper semantic memory search (replace keyword matching)
- [ ] Voice input/output
- [ ] Vision support
- [ ] A better frontend
- [ ] Swap the Ollama backend for a remote API without breaking everything else
- [ ] Model selection from the UI

---

## Status

I'm stepping back from this for now. The core architecture works the way I intended, but the response quality from locally-run small models hasn't been good enough for daily use — and that's more a hardware/model limitation than a code problem. I may pick it up again later.

---

## License

MIT — do whatever you want with it.

---

## Author

Built by [LeCaseus](https://github.com/LeCaseus) as a personal learning project.
