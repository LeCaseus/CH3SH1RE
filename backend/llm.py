import requests
import json
from .prompts import get_thinking_prompt

# Ollama runs locally on this port — no API key, no cloud
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"


def ask_llm(messages: list, stream: bool = False) -> str:
    """Send a conversation to Ollama and return the response.

    messages: full conversation history as list of role/content dicts
    stream: True only for final answers shown to user, False for internal calls
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": "1h",
    }

    if stream:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        full_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except:
                continue
            chunk = data.get("message", {}).get("content", "")
            print(chunk, end="", flush=True)
            full_content += chunk
            if data.get("done"):
                break
        print()
        return full_content.strip()
    else:
        response = requests.post(OLLAMA_URL, json=payload)
        data = response.json()
        return data.get("message", {}).get("content", "").strip()


def think(messages: list) -> str:
    """Silent reasoning pass — output never shown to user.
    Takes the same message list as ask_llm, appends a thinking instruction,
    and returns the model's raw reasoning as a string."""
    original_system_content = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )

    thinking_prompt = get_thinking_prompt()
    thinking_prompt[
        "content"
    ] += f"\n\nContext you have available:\n{original_system_content}"

    thinking_messages = [thinking_prompt] + [
        m for m in messages if m["role"] != "system"
    ]
    thinking_messages.append(
        {
            "role": "user",
            "content": "Think through the question above step by step before I ask you to answer.",
        }
    )
    return ask_llm(thinking_messages, stream=False)


def stream_llm_chunks(messages: list):
    """Generator that yields raw content chunks from Ollama one at a time.
    Used by the /chat endpoint so the frontend can render words as they arrive
    instead of waiting for the full response."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "1h",
    }
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        chunk = data.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if data.get("done"):
            break
