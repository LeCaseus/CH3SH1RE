import json
from .llm import ask_llm
from .tools import search_web

# LLM decides after each round whether results are sufficient,
# or what to search next — it never sees this prompt in the final answer
_SUFFICIENCY_PROMPT = (
    "You are a research planner. You have a user question and accumulated search results. "
    "Decide whether the results are sufficient to answer the question thoroughly. "
    "Return ONLY a raw JSON object — no markdown, no explanation, nothing else. "
    'If sufficient: {"sufficient": true} '
    'If not: {"sufficient": false, "next_query": "the next search query to run"} '
    "Rules: "
    "1. next_query must be meaningfully different from all previous queries — not a rephrasing. "
    "2. next_query should go deeper or fill a specific gap, not repeat what was already searched. "
    "3. Be strict — vague or partial answers are not sufficient. "
    '4. If you cannot identify a useful next query, return {"sufficient": true} to stop the loop.'
)

# how many search rounds before we stop regardless of sufficiency
MAX_ROUNDS = 4


def deep_research(question: str) -> str:
    """Run a multi-round search loop and return all accumulated results as a string.

    Each round:
      1. Run a web search with the current query
      2. Ask the LLM: enough to answer, or what to search next?
      3. Stop if sufficient, hit MAX_ROUNDS, or no valid next query returned

    The returned string is injected into the system prompt as search context —
    the calling code handles synthesis via get_search_synthesis_prompt().
    """
    all_results: list[str] = []
    previous_queries: list[str] = []
    query = question

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"[deep research] round {round_num} — query: '{query}'")

        results = search_web(query)
        all_results.append(f"[Round {round_num} | Query: {query}]\n{results}")
        previous_queries.append(query)

        # hit the cap — stop here and use what we have
        if round_num == MAX_ROUNDS:
            print(f"[deep research] reached {MAX_ROUNDS}-round limit, synthesising.")
            break

        # ask LLM if we have enough or what to look for next
        check_messages = [
            {"role": "system", "content": _SUFFICIENCY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Queries already run: {', '.join(previous_queries)}\n\n"
                    f"Search results so far:\n\n" + "\n\n".join(all_results)
                ),
            },
        ]

        raw = ask_llm(check_messages, stream=False)

        try:
            # strip markdown fences in case the model wraps the JSON anyway
            clean = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            decision = json.loads(clean)
        except Exception:
            # malformed response — stop and synthesise what we have
            print("[deep research] sufficiency check returned unparseable output, stopping.")
            break

        if decision.get("sufficient"):
            print(f"[deep research] sufficient after {round_num} round(s).")
            break

        next_query = decision.get("next_query", "").strip()
        if not next_query or next_query in previous_queries:
            # no useful next step — stop rather than loop pointlessly
            print("[deep research] no new query returned, stopping.")
            break

        query = next_query

    return "\n\n".join(all_results)
