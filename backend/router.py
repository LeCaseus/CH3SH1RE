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
from .memory import get_all_facts
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
            "job application",
            "applying for",
            "application for",
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
            "recommend a recipe",
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

# quick lookups — one search, answer immediately
# these are time-sensitive or factual point queries that don't benefit from iteration
QUICK_SEARCH_TRIGGERS = [
    "today",
    "right now",
    "weather",
    "price",
    "score",
    "news",
    "latest",
    "current",
    "now",
    "update",
    "recent",
    "happened",
]

# deep research triggers — questions that require synthesis across multiple sources
# these are exploratory or analytical, not point-in-time lookups
DEEP_SEARCH_TRIGGERS = [
    "how does",
    "how do",
    "why does",
    "why do",
    "what is",
    "what are",
    "explain",
    "research",
    "deep dive",
    "tell me about",
    "find out",
    "look into",
    "investigate",
    "overview of",
    "breakdown of",
    "history of",
    "impact of",
    "effect of",
    "benefits of",
    "risks of",
    "pros and cons of",
    "everything about",
    "what should i know",
    "help me understand",
]


def detect_intent(user_input: str) -> tuple[str, Callable]:
    """Returns (intent_name, prompt_fn)."""
    text = user_input.lower()

    for intent, prompt_fn, keywords in INTENT_MAP:
        if any(kw in text for kw in keywords):
            return intent, prompt_fn

    return "chat", get_chat_prompt


def needs_quick_search(user_input: str) -> bool:
    """True for time-sensitive or point-in-time lookups — runs one search only."""
    text = user_input.lower()
    return any(trigger in text for trigger in QUICK_SEARCH_TRIGGERS)


def needs_deep_research(user_input: str) -> bool:
    """True for exploratory or analytical questions — runs the multi-round loop.
    Only fires if the query is NOT already caught by needs_quick_search,
    so "latest news about how vaccines work" stays a quick lookup."""
    text = user_input.lower()
    if len(text.split()) < 7:
        return False
    if needs_quick_search(user_input):
        return False
    return any(trigger in text for trigger in DEEP_SEARCH_TRIGGERS)


def build_messages(
    user_input: str, memories: list[dict], file_path: Optional[str] = None
) -> list[dict]:
    # imported here to avoid circular import at module level
    from .llm import think
    from .researcher import deep_research

    intent_name, prompt_fn = detect_intent(user_input)

    file_context = ""
    if file_path:
        file_context = f"\n\nDocument provided by user:\n{read_file(file_path)}"

    search_context = ""
    search_fired = False

    # quick search takes priority — if the query is time-sensitive, don't loop
    if needs_quick_search(user_input):
        print("Quick search...")
        results = search_web(user_input)
        search_context = f"\n\nWeb search results:\n{results}"
        prompt_fn = get_search_synthesis_prompt
        search_fired = True

    # deep research fires only when the query is exploratory, not a quick lookup
    elif needs_deep_research(user_input):
        print("Starting deep research...")
        results = deep_research(user_input)
        search_context = f"\n\nResearch results (multiple rounds):\n{results}"
        prompt_fn = get_search_synthesis_prompt
        search_fired = True

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

    # inject known user facts and behavioural instructions first,
    # so the model treats them as ground truth before anything else
    facts = get_all_facts()
    if facts["facts"] or facts["instructions"]:
        prefix = ""
        if facts["facts"]:
            prefix += f"Known facts about this user:\n{facts['facts']}\n\n"
        if facts["instructions"]:
            prefix += f"Instructions for responding to this user:\n{facts['instructions']}\n\n"
        system["content"] = prefix + system["content"]

    base_messages = [system, {"role": "user", "content": user_input}]

    # reasoning pass for complex intents — runs after search so the model
    # can reason over actual results, not just the raw question
    if intent_name in COT_INTENTS and not search_fired:
        print(f"Thinking [{intent_name}]...")
        reasoning = think(base_messages)
        system["content"] += (
            f"\n\nYour internal reasoning (use this to inform your answer, "
            f"do not repeat it verbatim):\n{reasoning}"
        )

    return [system, {"role": "user", "content": user_input}]
