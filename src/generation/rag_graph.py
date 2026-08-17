"""
rag_graph.py

Models the full RAG pipeline as an explicit LangGraph StateGraph:

    retrieve -> rerank -> expand_context -> generate

Why a graph instead of one function calling the next directly:
  - Each step is independently testable and swappable (e.g. later we can
    insert a multi_query or query_expansion node before `retrieve`, or a
    citation_validate node after `generate`, without restructuring
    everything else).
  - LangGraph gives us built-in state tracking — every node reads and
    writes to one shared, typed state object, so it's easy to see exactly
    what data exists at each point in the pipeline (useful for debugging
    AND for explaining the system in an interview).
  - This is also the foundation Phase 3's agentic query decomposition
    would build on — an agent node can just be another node in this graph
    that conditionally routes to sub-graphs.

Usage (from project root, Qdrant running + GROQ_API_KEY set):
    python -m src.generation.rag_graph "how do I create an issue"
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.retrieval.fusion import hybrid_search
from src.reranking.cross_encoder import rerank
from src.retrieval.parent_child import expand_to_parent, format_for_generation
from src.generation.generator import generate_answer
from src.generation.citation_validator import validate_citations, strip_citation_tags
from src.generation.conversation_manager import condense_query
from src.retrieval.query_expansion import expand_query

MAX_REGENERATION_ATTEMPTS = 1  # only retry once — avoid infinite loops on a persistently bad model


class RAGState(TypedDict):
    query: str                    # the raw, as-typed question for this turn
    chat_history: list[dict]      # prior {'query':..., 'answer':...} turns, oldest first
    standalone_query: str         # query rewritten to be self-contained, used for generation
    search_query: str              # standalone_query + shorthand expansion, used for retrieval only
    retrieve_k: int
    rerank_top_k: int
    qdrant_url: str
    candidates: list[dict]        # after hybrid_search
    reranked: list[dict]          # after cross-encoder reranking
    expanded_endpoint_ids: set[str]  # ground truth for citation validation
    context_text: str             # after parent-child expansion + formatting
    answer: str                   # final LLM output (raw, with [cite: ...] tags)
    display_answer: str           # citation tags stripped, ready to show the user
    citation_valid: bool
    regeneration_attempts: int


def condense_query_node(state: RAGState) -> dict:
    standalone = condense_query(state["query"], state["chat_history"])
    # Expansion is for RETRIEVAL only — appending "pull request" after "pr"
    # helps BM25/vector matching, but we don't want that duplicated wording
    # bleeding into what we show the LLM as "the question" for generation.
    search_query = expand_query(standalone)
    return {"standalone_query": standalone, "search_query": search_query}


def retrieve_node(state: RAGState) -> dict:
    candidates = hybrid_search(
        state["search_query"], top_k=state["retrieve_k"], qdrant_url=state["qdrant_url"]
    )
    return {"candidates": candidates}


def rerank_node(state: RAGState) -> dict:
    reranked = rerank(state["search_query"], state["candidates"], top_k=state["rerank_top_k"])
    return {"reranked": reranked}


def expand_context_node(state: RAGState) -> dict:
    expanded = expand_to_parent(state["reranked"])
    context_text = format_for_generation(expanded)
    endpoint_ids = {c["endpoint_id"] for c in expanded}
    return {"context_text": context_text, "expanded_endpoint_ids": endpoint_ids}


def generate_node(state: RAGState) -> dict:
    query = state["standalone_query"]
    context_text = state["context_text"]

    # On a retry (after a hallucinated citation was caught), add a stronger
    # correction note rather than silently re-asking the same question —
    # gives the model a real chance to fix the specific failure mode.
    if state.get("regeneration_attempts", 0) > 0:
        query = (
            f"{query}\n\n(Note: your previous answer cited an endpoint that does not appear "
            f"in the documentation context below. Only use [cite: ...] tags with endpoint_ids "
            f"that literally match a '===' section header in the context.)"
        )

    answer = generate_answer(query, context_text)
    return {"answer": answer}


def validate_citations_node(state: RAGState) -> dict:
    result = validate_citations(state["answer"], state["expanded_endpoint_ids"])
    return {
        "citation_valid": result.is_valid,
        "display_answer": strip_citation_tags(state["answer"]) if result.is_valid else state["answer"],
    }


def _route_after_validation(state: RAGState) -> str:
    """Conditional edge: regenerate once on a hallucinated citation, otherwise stop."""
    if state["citation_valid"]:
        return "end"
    if state.get("regeneration_attempts", 0) >= MAX_REGENERATION_ATTEMPTS:
        # give up after one retry — surface the answer anyway rather than loop forever,
        # but the raw (unstripped) answer stays visible so the issue isn't hidden.
        return "end"
    return "regenerate"


def increment_retry_node(state: RAGState) -> dict:
    return {"regeneration_attempts": state.get("regeneration_attempts", 0) + 1}


def build_rag_graph():
    """Compile the StateGraph. Call once, reuse the compiled graph across queries."""
    graph = StateGraph(RAGState)

    graph.add_node("condense_query", condense_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("expand_context", expand_context_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate_citations", validate_citations_node)
    graph.add_node("increment_retry", increment_retry_node)

    graph.set_entry_point("condense_query")
    graph.add_edge("condense_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "expand_context")
    graph.add_edge("expand_context", "generate")
    graph.add_edge("generate", "validate_citations")
    graph.add_conditional_edges(
        "validate_citations",
        _route_after_validation,
        {"end": END, "regenerate": "increment_retry"},
    )
    graph.add_edge("increment_retry", "generate")

    return graph.compile()


def run_rag_pipeline(
    query: str,
    chat_history: list[dict] | None = None,
    retrieve_k: int = 25,
    rerank_top_k: int = 3,
    qdrant_url: str = "http://localhost:6333",
) -> RAGState:
    """
    Convenience wrapper: build the graph fresh and run one query through it.

    If LangSmith tracing is enabled (LANGSMITH_TRACING=true + LANGSMITH_API_KEY
    set — see README "Observability" section), this run is automatically
    traced. The run_name/tags/metadata below exist so a trace in the
    LangSmith UI reads as "ask-my-docs-query" with the actual question
    visible at a glance, instead of a generic "LangGraph" label you'd have
    to click into to identify.
    """
    app = build_rag_graph()
    initial_state: RAGState = {
        "query": query,
        "chat_history": chat_history or [],
        "standalone_query": "",
        "search_query": "",
        "retrieve_k": retrieve_k,
        "rerank_top_k": rerank_top_k,
        "qdrant_url": qdrant_url,
        "candidates": [],
        "reranked": [],
        "expanded_endpoint_ids": set(),
        "context_text": "",
        "answer": "",
        "display_answer": "",
        "citation_valid": False,
        "regeneration_attempts": 0,
    }
    trace_config = {
        "run_name": "ask-my-docs-query",
        "tags": ["conversational" if chat_history else "single-turn"],
        "metadata": {
            "query": query,
            "has_history": bool(chat_history),
            "retrieve_k": retrieve_k,
            "rerank_top_k": rerank_top_k,
        },
    }
    final_state = app.invoke(initial_state, config=trace_config)
    return final_state


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run the full LangGraph RAG pipeline for one query.")
    ap.add_argument("query")
    ap.add_argument("--retrieve-k", type=int, default=25)
    ap.add_argument("--rerank-top-k", type=int, default=3)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--show-context", action="store_true", help="Print the assembled context before the answer")
    args = ap.parse_args()

    print(f"Running RAG pipeline for: {args.query!r}\n")
    result = run_rag_pipeline(
        args.query,
        retrieve_k=args.retrieve_k,
        rerank_top_k=args.rerank_top_k,
        qdrant_url=args.qdrant_url,
    )

    if result["standalone_query"] != result["query"]:
        print(f"(condensed to standalone query: {result['standalone_query']!r})\n")
    if result["search_query"] != result["standalone_query"]:
        print(f"(expanded for search: {result['search_query']!r})\n")

    if args.show_context:
        print("=" * 70)
        print("CONTEXT SENT TO LLM:")
        print("=" * 70)
        print(result["context_text"][:2000])
        print()

    print("=" * 70)
    print("ANSWER:")
    print("=" * 70)
    print(result["display_answer"])
    print()
    print(f"[citation check: {'PASSED' if result['citation_valid'] else 'FAILED after retry'} "
          f"| regeneration attempts: {result['regeneration_attempts']}]")


if __name__ == "__main__":
    main()