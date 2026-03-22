from .prompts import (
    get_chat_prompt,
    get_writing_prompt,
    get_summarise_prompt,
    get_research_prompt,
    get_study_prompt,
    get_planning_prompt,
    get_career_prompt,
    get_personal_life_prompt,
    get_search_synthesis_prompt,
)
from .tools import search_web, read_file
from .llm import ask_llm
from typing import Callable, Optional

# intents complex enough to benefit from a reasoning pass
COT_INTENTS = {"career", "research", "study", "planning"}

# keywords that trigger each prompt — order matters, more specific first
INTENT_MAP = [
    (
        "career",
        get_career_prompt,
        [
            "cv",
            "resume",
            "cover letter",
            "job",
            "interview",
            "linkedin",
            "salary",
            "hire",
            "hiring",
            "application",
            "fit for",
            "apply",
        ],
    ),
    (
        "writing",
        get_writing_prompt,
        [
            "write",
            "draft",
            "rewrite",
            "proofread",
            "rephrase",
            "tone",
            "email",
            "caption",
            "bio",
            "grammar",
            "formal",
            "casual",
        ],
    ),
    (
        "summarise",
        get_summarise_prompt,
        [
            "summarise",
            "summarize",
            "summary",
            "tldr",
            "key points",
            "shorten",
            "condense",
            "brief",
        ],
    ),
    (
        "study",
        get_study_prompt,
        [
            "flashcard",
            "quiz",
            "study",
            "teach me",
            "test me",
            "definition",
            "concept",
            "explain to me",
        ],
    ),
    (
        "research",
        get_research_prompt,
        [
            "compare",
            "difference between",
            "which is better",
            "pros and cons",
            "recommend",
            "should i",
            "best",
            "vs",
            "versus",
            "review",
        ],
    ),
    (
        "planning",
        get_planning_prompt,
        [
            "plan",
            "schedule",
            "meal",
            "trip",
            "itinerary",
            "todo",
            "to-do",
            "brainstorm",
            "decide",
            "prioritise",
            "prioritize",
            "organise",
            "organize",
            "ideas for",
        ],
    ),
    (
        "personal",
        get_personal_life_prompt,
        [
            "recipe",
            "cook",
            "ingredient",
            "relationship",
            "feel",
            "feeling",
            "anxious",
            "stressed",
            "budget",
            "money",
            "health",
            "symptom",
            "doctor",
            "advice",
            "vent",
        ],
    ),
]

# keywords that trigger a web search
SEARCH_TRIGGERS = [
    "today",
    "latest",
    "current",
    "news",
    "now",
    "price",
    "weather",
    "score",
    "update",
    "recent",
    "right now",
    "happened",
]


def detect_intent(user_input: str) -> tuple[str, Callable]:
    """Returns (intent_name, prompt_fn)."""
    text = user_input.lower()

    for intent, prompt_fn, keywords in INTENT_MAP:
        if any(kw in text for kw in keywords):
            return intent, prompt_fn

    # default to general chat
    return "chat", get_chat_prompt


def needs_search(user_input: str) -> bool:
    """Decide if the query needs a live web search."""
    text = user_input.lower()
    return any(trigger in text for trigger in SEARCH_TRIGGERS)


def build_messages(
    user_input: str, memories: list[dict], file_path: Optional[str] = None
) -> list[dict]:
    from .llm import think  # here to avoid circular import at module level

    intent_name, prompt_fn = detect_intent(user_input)

    file_context = ""
    if file_path:
        file_context = f"\n\nDocument provided by user:\n{read_file(file_path)}"

    search_context = ""
    if needs_search(user_input):
        print("Searching the web...")
        results = search_web(user_input)
        search_context = f"\n\nWeb search results:\n{results}"
        prompt_fn = get_search_synthesis_prompt

    memory_context = ""
    if memories:
        pairs = []
        for i in range(0, len(memories) - 1, 2):
            user_msg = memories[i]["content"]
            ai_msg = memories[i + 1]["content"] if i + 1 < len(memories) else ""
            pairs.append(f"User: {user_msg}\nAssistant: {ai_msg}")
        memory_context = (
            "\n\nPast conversation context (for reference only, do not repeat or re-answer):\n"
            + "\n---\n".join(pairs)
        )

    system = prompt_fn()
    system["content"] += memory_context + file_context + search_context

    base_messages = [system, {"role": "user", "content": user_input}]

    # reasoning pass for complex intents
    if intent_name in COT_INTENTS:
        print(f"Thinking [{intent_name}]...")
        reasoning = think(base_messages)
        system["content"] += (
            f"\n\nYour internal reasoning (use this to inform your answer, "
            f"do not repeat it verbatim):\n{reasoning}"
        )

    return [system, {"role": "user", "content": user_input}]
