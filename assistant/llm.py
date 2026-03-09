import requests
import json

LLM_URL = "http://localhost:8080/v1/chat/completions"


def ask_llm(prompt, memories=None, stream_print=True):
    system_prompt = """You are a reasoning planner with tool access.

        You MUST follow this decision pipeline:

        1. Decide if you already know the answer.
        2. If information is required → call exactly ONE tool.
           - Only use a tool for **informational** questions or when the
             conversation history suggests a factual lookup. Do not invoke
             any tool for simple conversational utterances like greetings,
             farewells, or short acknowledgments.
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
        """

    memory_context = ""
    if memories:
        memory_context = "Recent history:\n" + "\n".join(memories) + "\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": memory_context + prompt}
    ]

    payload = {
        "messages": messages,
        "max_tokens": 512,
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
            chunk = data.get("choices", [{}])[0].get("delta", {}).get("content")
            if chunk is None:
                continue
            chunk = str(chunk)
            print(chunk, end="", flush=True)
            full_content += chunk
            if data.get("choices", [{}])[0].get("finish_reason"):
                break
        print()
        return str(full_content.strip())
    else:
        response = requests.post(LLM_URL, json=payload)
        json_data = response.json()

        if not json_data:
            return "No response from model"

        if "choices" not in json_data or len(json_data["choices"]) == 0:
            return "No valid completion returned"

        content = json_data["choices"][0].get("message", {}).get("content")

        if content is None:
            return "Empty model response"

        return str(content).strip()
