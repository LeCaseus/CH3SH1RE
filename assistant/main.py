from .memory import init_memory_db, get_recent_memories, save_memory, search_semantic
from .llm import ask_llm
from .router import handle_tool, extract_json

def main():
    init_memory_db()
    print("Local AI Assistant Started")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        memories = get_recent_memories(limit=3)
        semantic_memories = search_semantic(user_input, top_k=2)
        memories = memories + semantic_memories
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
                context = (
                    f"Original question: {user_input}\n\nCollected information so far:\n"
                    + "\n".join(collected_info)
                    + f"\n\nLatest result: {tool_result}\n\nIf you need more information, use tools. Otherwise, provide a comprehensive final answer."
                )
            else:
                # No tool needed or final answer ready
                final_answer = response
                break

        # If we have collected information but no final answer, synthesize one
        if not final_answer and collected_info:
            synthesis_prompt = (
                f"Original question: {user_input}\n\nAll collected information:\n"
                + "\n".join(collected_info)
                + "\n\nBased on all the information gathered, provide a comprehensive final answer."
            )
            final_answer = ask_llm(synthesis_prompt, memories, stream_print=False)

        print("AI:", final_answer)
        save_memory(user_input, final_answer)


if __name__ == "__main__":
    main()
