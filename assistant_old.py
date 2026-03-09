import requests
import sqlite3
import datetime
import json
from bs4 import BeautifulSoup
from typer import prompt

# -----------------------------
# Memory Database
# -----------------------------

def init_memory_db():
    conn = sqlite3.connect('assistant_memory.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS memories
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  user_input TEXT,
                  ai_response TEXT,
                  summary TEXT)''')
    conn.commit()
    conn.close()

def save_memory(user_input, ai_response):
    summary = f"User: {user_input[:50]}... AI: {ai_response[:50]}..."
    conn = sqlite3.connect('assistant_memory.db')
    c = conn.cursor()
    c.execute("INSERT INTO memories (timestamp, user_input, ai_response, summary) VALUES (?, ?, ?, ?)",
              (datetime.datetime.now().isoformat(), user_input, ai_response, summary))
    conn.commit()
    conn.close()

def get_recent_memories(limit=5):
    conn = sqlite3.connect('assistant_memory.db')
    c = conn.cursor()
    c.execute("SELECT summary FROM memories ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows[::-1]]  # Reverse to chronological order

LLM_URL = "http://localhost:8080/v1/chat/completions"

def ask_llm(prompt, memories=None, stream_print=True):
    system_prompt = """<|im_start|>system
        You are a reasoning planner with tool access.

        You MUST follow this decision pipeline:

        1. Decide if you already know the answer.
        2. If information is required → call exactly ONE tool.
        3. After receiving tool results → either:
        - Provide final answer
        - Or request another tool

        Strict rules:
        - Never invent facts.
        - Never mix tool JSON with explanation text.
        - If search results are empty → state information was not found.
        - Do not repeat tool calls after producing final answer.
        - Prefer factual verification over guessing.

        Tool formats:

        Search:
        {"tool":"search","parameters":{"query":"search query"}}

        Image:
        {"tool":"image","parameters":{"path":"file path"}}

        If no tool is needed:
        {"tool":"none"}
        <|im_end|>
        """

    memory_context = ""
    if memories:
        memory_context = "<|im_start|>user\nRecent history:\n" + "\n".join(memories) + "\n<|im_end|>"

    full_prompt = f"""<|im_start|>system
        {system_prompt}
        {memory_context}
        <|im_end|>

        <|im_start|>user
        {prompt}
        <|im_end|>

        <|im_start|>assistant
        """

    payload = {
        "prompt": full_prompt,
        "n_predict": 512,
        "temperature": 0.7,
        "stream": stream_print
    }

    if stream_print:
        response = requests.post(LLM_URL, json=payload, stream=True)
        full_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data:"):
                continue
            decoded = decoded.replace("data:", "").strip()
            if decoded == "[DONE]":
                break
            try:
                data = json.loads(decoded)
            except:
                continue
            chunk = data.get("content", "")
            print(chunk, end="", flush=True)
            full_content += chunk
            if data.get("stop", False):
                break
        print()
        return full_content.strip()
    else:
        response = requests.post(LLM_URL, json=payload)
        return response.json()["content"]

# -----------------------------
# Tool Functions
# -----------------------------

def search_web(query):
    """
    Search using DuckDuckGo HTML Query Scraping.
    Uses the HTML endpoint with proper headers and parsing.
    """
    try:
        url = "https://duckduckgo.com/html/"
        
        # DuckDuckGo requires a User-Agent header
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        params = {"q": query}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")

        results = []

        # Extract all results from DuckDuckGo HTML - each result is in a .result div
        for result in soup.select("div.result"):
            # Title is in a span with class result__title inside an a tag
            title_elem = result.select_one("a.result__url")
            snippet_elem = result.select_one("a.result__snippet")
            
            # Alternative selectors for robustness
            if not title_elem:
                title_elem = result.select_one(".result__title")
            if not snippet_elem:
                snippet_elem = result.select_one(".result__snippet")

            if not title_elem or not snippet_elem:
                continue

            title_text = title_elem.get_text(strip=True)
            snippet_text = snippet_elem.get_text(strip=True)
            url_text = title_elem.get("href", "")

            # Skip if URL is relative or malformed
            if not url_text or not url_text.startswith("http"): # type: ignore
                continue

            # Filter out low-quality results (empty or very short snippets)
            if len(snippet_text) < 20:
                continue

            results.append({
                "title": title_text,
                "url": url_text,
                "snippet": snippet_text
            })

            # Stop after getting 3 strong results
            if len(results) >= 3:
                break

        if results:
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. {r['title']}\n"
                    f"   {r['snippet']}\n"
                    f"   ({r['url']})"
                )
            return "\n\n".join(formatted)

        return f"No quality results found for '{query}'"

    except requests.exceptions.RequestException as e:
        return f"Search error: Failed to contact DuckDuckGo - {str(e)}"
    except Exception as e:
        return f"Search error: {str(e)}"

def analyze_image(image_path):
    # Calls the separate vision service running on port 5000
    # Start vision_service.py separately before using image analysis
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://127.0.0.1:5000/analyze', files=files, timeout=10)
            data = response.json()
            return data.get('description', 'Unable to analyze image')
    except Exception as e:
        return f"Image analysis error: {str(e)}"


# -----------------------------
# Tool Router
# -----------------------------

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
    
# -----------------------------
# Main Chat Loop
# -----------------------------

def main():
    init_memory_db()
    print("Local AI Assistant Started")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        memories = get_recent_memories()
        print("Processing user input...")

        # Planning and execution loop (max 3 steps)
        context = user_input
        collected_info = []
        final_answer = ""
        max_steps = 3

        for step in range(max_steps):
            print(f"Planning step {step + 1}...")

            # Get response from LLM with current context
            response = ask_llm(context, memories)

            # Check if response contains a tool call
            tool_call = extract_json(response)

            if tool_call and tool_call.get("tool") not in ["none", None]:
                # Execute the tool
                tool_result = handle_tool(tool_call, user_input, memories)
                collected_info.append(f"Step {step + 1} result: {tool_result}")

                # Update context for next step
                context = f"Original question: {user_input}\n\nCollected information so far:\n" + "\n".join(collected_info) + f"\n\nLatest result: {tool_result}\n\nIf you need more information, use tools. Otherwise, provide a comprehensive final answer."
            else:
                # No tool needed or final answer ready
                final_answer = response
                break

        # If we have collected information but no final answer, synthesize one
        if not final_answer and collected_info:
            synthesis_prompt = f"Original question: {user_input}\n\nAll collected information:\n" + "\n".join(collected_info) + "\n\nBased on all the information gathered, provide a comprehensive final answer."
            final_answer = ask_llm(synthesis_prompt, memories, stream_print=False)

        print("AI:", final_answer)
        save_memory(user_input, final_answer)

if __name__ == "__main__":
    main()