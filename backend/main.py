from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from contextlib import asynccontextmanager
from pathlib import Path

import shutil
import os

from .memory import init_db, save_memory, get_recent_memories, search_memories
from .router import build_messages
from .llm import ask_llm, stream_llm_chunks

app = FastAPI()

# allows the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent  # points to CH3SH1RE/


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs on startup
    init_db()
    ask_llm([{"role": "user", "content": "hello"}])
    print("CH3SH1RE is running.")
    yield
    # anything after yield runs on shutdown (nothing needed for now)


app = FastAPI(lifespan=lifespan)


@app.get("/")
def serve_frontend():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


@app.get("/static/style.css")
def serve_css():
    return FileResponse(str(BASE_DIR / "frontend" / "style.css"))


@app.get("/static/app.js")
def serve_js():
    return FileResponse(str(BASE_DIR / "frontend" / "app.js"))


@app.post("/chat")
def chat(message: str = Form(...), file: UploadFile = File(None)):
    # save uploaded file if provided
    file_path = None
    if file and file.filename:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"File uploaded: {file_path}")

    # get recent memory + semantic search for relevant past context
    recent = get_recent_memories(limit=6)
    relevant = search_memories(message, limit=2)

    # deduplicate — avoid sending same exchange twice
    seen = set()
    memories = []
    for m in recent + relevant:
        key = m["content"]
        if key not in seen:
            seen.add(key)
            memories.append(m)

    # build full message list
    messages = build_messages(message, memories, file_path)

    def generate():
        # stream chunks to the browser as they arrive from Ollama
        # accumulate the full response so we can save it to memory at the end
        full_response = ""
        for chunk in stream_llm_chunks(messages):
            full_response += chunk
            yield chunk
        save_memory(message, full_response)

    return StreamingResponse(generate(), media_type="text/plain")
