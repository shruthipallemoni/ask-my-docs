"""
generator.py

Calls Groq's free API tier to generate an answer from retrieved context —
via LangChain's ChatGroq wrapper, so this slots directly into the LangGraph
pipeline (rag_graph.py) as one node among several.

SECURITY NOTE: the Groq API key is read from the GROQ_API_KEY environment
variable — never hardcode it in source, never commit it, never paste it
into a chat. Set it once per terminal session:

    Windows (PowerShell):  $env:GROQ_API_KEY = "your-key-here"
    Mac/Linux:              export GROQ_API_KEY="your-key-here"

Model choice: openai/gpt-oss-20b — Groq's fastest model, with the most
free-tier headroom (tokens-per-minute) of the strong general-purpose
options. gpt-oss-120b is a good Phase 3 quality comparison, but its
tighter free-tier token budget makes it easy to hit rate limits during
active development.

Usage (from project root, GROQ_API_KEY set):
    python -m src.generation.generator "how do I create an issue"
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt

MODEL_NAME = "openai/gpt-oss-20b"

_llm: ChatGroq | None = None


def _get_llm(temperature: float = 0.1) -> ChatGroq:
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY environment variable not set. "
                "Get a free key at https://console.groq.com/keys and set it with:\n"
                '  PowerShell: $env:GROQ_API_KEY = "your-key-here"\n'
                '  Mac/Linux:  export GROQ_API_KEY="your-key-here"'
            )
        # temperature kept low on purpose — this is a factual documentation
        # assistant, not creative writing. Lower temperature = less likely
        # to embellish beyond the provided context.
        _llm = ChatGroq(model=MODEL_NAME, temperature=temperature, max_tokens=1024)
    return _llm


def generate_answer(query: str, context_text: str, temperature: float = 0.1) -> str:
    """
    Generate a citation-enforced answer given the query and the assembled
    documentation context (typically the output of parent_child.format_for_generation).
    """
    llm = _get_llm(temperature=temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(query, context_text)),
    ]
    response = llm.invoke(messages)
    return response.content


def main():
    import argparse
    from src.reranking.cross_encoder import rerank
    from src.retrieval.fusion import hybrid_search
    from src.retrieval.parent_child import expand_to_parent, format_for_generation

    ap = argparse.ArgumentParser(description="Run the full pipeline: retrieve, rerank, expand, generate.")
    ap.add_argument("query")
    ap.add_argument("--retrieve-k", type=int, default=25)
    ap.add_argument("--rerank-top-k", type=int, default=3)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    print("Retrieving...")
    candidates = hybrid_search(args.query, top_k=args.retrieve_k, qdrant_url=args.qdrant_url)

    print("Reranking...")
    reranked = rerank(args.query, candidates, top_k=args.rerank_top_k)

    print("Expanding to full endpoint docs...")
    expanded = expand_to_parent(reranked)
    context_text = format_for_generation(expanded)

    print(f"Generating answer with {MODEL_NAME}...\n")
    answer = generate_answer(args.query, context_text)

    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()