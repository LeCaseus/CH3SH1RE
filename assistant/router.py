import json
from .tools import search_web, analyze_image


def handle_tool(tool_call, user_input, memories=None):
    """Execute a tool call and return the raw result."""
    tool = tool_call.get("tool")
    params = tool_call.get("parameters", {})

    if tool == "search":
        query = params.get("query", "")
        print(f"Assistant detected need for web search: {query}")
        print("Querying search service...")

        results = search_web(query)
        print(f"Search results retrieved ({len(results)} chars)")

        return f"Search results for '{query}':\n{results}"

    elif tool == "image":
        path = params.get("path", "")
        print(f"Assistant detected need for vision analysis of: {path}")
        print("Requesting vision service...")

        results = analyze_image(path)
        print(f"Vision analysis complete: {results}")

        return f"Image analysis of '{path}':\n{results}"

    return "Unknown tool requested"


def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            return None

        return json.loads(text[start:end+1])

    except:
        return None
