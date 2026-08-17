"""
chat_session.py

Interactive multi-turn REPL for testing conversational RAG. Keeps a
growing chat_history across turns and passes it into the pipeline each
time, so follow-up questions like "what about DELETE?" get condensed into
standalone queries using prior turns as context.

Usage (from project root, Qdrant running + GROQ_API_KEY set):
    python -m src.generation.chat_session

Then just type questions. Type 'exit' or 'quit' to stop.
"""

from src.generation.rag_graph import run_rag_pipeline


def main():
    print("Ask My Docs — interactive session. Type 'exit' to quit.\n")
    chat_history: list[dict] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        result = run_rag_pipeline(query, chat_history=chat_history)

        if result["standalone_query"] != result["query"]:
            print(f"  (understood as: {result['standalone_query']!r})")
        if result["search_query"] != result["standalone_query"]:
            print(f"  (expanded for search: {result['search_query']!r})")

        print(f"\nAssistant: {result['display_answer']}\n")

        if not result["citation_valid"]:
            print("  [warning: this answer's citations could not be fully verified]\n")

        # Store the raw query + final display answer, not the standalone
        # rewrite — future condensation looks at what was actually said in
        # the conversation, not our internal rewriting.
        chat_history.append({"query": query, "answer": result["display_answer"]})


if __name__ == "__main__":
    main()