import requests
import json
from concurrent.futures import ThreadPoolExecutor
from .prompts import get_thinking_prompt, get_synthesis_prompt

# Ollama runs locally on this port — no API key, no cloud
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

# temperature presets for multi-temperature synthesis
TEMP_CAUTIOUS = 0.3  # grounded, conservative — sticks close to facts
TEMP_CREATIVE = 0.8  # expansive, associative — surfaces angles the cautious pass misses


def ask_llm(
    messages: list, stream: bool = False, temperature: float | None = None
) -> str:
    """Send a conversation to Ollama and return the response.

    messages:    full conversation history as list of role/content dicts
    stream:      True only for final answers shown to user, False for internal calls
    temperature: overrides model default when set — None leaves Ollama's default alone
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "keep_alive": "1h",
    }

    # only inject temperature when explicitly set — None means leave Ollama's default alone
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    if stream:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        full_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
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


def synthesize(messages: list) -> str:
    """Multi-temperature synthesis — silent, never shown to user directly.

    The cautious and creative drafts are fired in parallel via ThreadPoolExecutor,
    so the wait is roughly one inference time instead of two sequential ones.
    A third synthesis pass then merges both drafts into a single best answer.

    Returns the synthesized string — router.py injects it into the system prompt
    so stream_llm_chunks() delivers the final response to the user.
    """
    print("[synthesis] cautious + creative drafts (parallel)...")

    # both drafts are independent — run them at the same time
    with ThreadPoolExecutor(max_workers=2) as executor:
        cautious_future = executor.submit(ask_llm, messages, False, TEMP_CAUTIOUS)
        creative_future = executor.submit(ask_llm, messages, False, TEMP_CREATIVE)
        cautious_draft = cautious_future.result()
        creative_draft = creative_future.result()

    user_question = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    print("[synthesis] synthesis pass...")
    synthesis_messages = [
        get_synthesis_prompt(),
        {
            "role": "user",
            "content": (
                f"Original question: {user_question}\n\n"
                f"Cautious draft (factual, conservative):\n{cautious_draft}\n\n"
                f"Creative draft (expansive, exploratory):\n{creative_draft}\n\n"
                "Write the best possible answer combining both."
            ),
        },
    ]

    # synthesis pass at default temperature — no need to skew either direction
    return ask_llm(synthesis_messages, stream=False)


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
