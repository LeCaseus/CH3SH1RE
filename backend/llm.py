import requests
import json

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
        print("DEBUG:", data)
        return data.get("message", {}).get("content", "").strip()
