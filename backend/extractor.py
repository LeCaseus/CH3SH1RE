import json
from .llm import ask_llm

# runs silently after every conversation to pull structured facts
# the model is told to be strict — no guessing, no fluff

_EXTRACTION_PROMPT = (
    "You extract structured facts about the user from a single conversation exchange. "
    "Return ONLY a raw JSON object — no markdown, no explanation, nothing else. "
    'Format: {"facts": [{"key": "snake_case_key", "value": "short fact"}]} '
    "Rules: "
    "1. Only extract facts clearly stated by the user, never inferred. "
    "2. Keys must be snake_case and stable (e.g. location, occupation, goal, tool_used). "
    "3. Values must be short and factual — one phrase, not a sentence. "
    '4. If nothing is extractable, return {"facts": []}.'
)


def extract_facts(user_input: str, ai_response: str) -> list[dict]:
    """Call the LLM to extract key/value facts from one exchange.
    Returns a list of {key, value} dicts, or [] if nothing found or parsing fails."""
    messages = [
        {"role": "system", "content": _EXTRACTION_PROMPT},
        {
            "role": "user",
            "content": (
                f"User said: {user_input}\n\n" f"Assistant replied: {ai_response}"
            ),
        },
    ]
    raw = ask_llm(messages, stream=False)

    try:
        # strip markdown fences if the model wraps the JSON anyway
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(clean)
        return data.get("facts", [])
    except Exception:
        # silently discard malformed output — facts are best-effort
        return []
