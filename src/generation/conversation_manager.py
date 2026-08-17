"""
conversation_manager.py

Handles multi-turn conversations by rewriting follow-up questions into
standalone queries BEFORE they hit retrieval. This is the piece that makes
"what about DELETE?" work as a follow-up to "how do I create an issue" —
without it, retrieval would search for the literal text "what about
DELETE?", which has almost no useful signal on its own.

Why condense before retrieval, not just feed raw history to the generator:
hybrid search (BM25 + vector) needs a self-contained query to work well.
"what about DELETE?" alone can't retrieve anything meaningful; "What does
the DELETE method do for /repos/{owner}/{repo}/issues?" can. So the
rewrite has to happen upstream of retrieval, not just at generation time.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

CONDENSE_MODEL = "openai/gpt-oss-20b"

CONDENSE_SYSTEM_PROMPT = """You rewrite a developer's follow-up question into a standalone question, using the conversation history for context. The rewritten question must make complete sense WITHOUT the history — someone reading only your output should understand exactly what's being asked.

Rules:
- If the follow-up question is already standalone (doesn't depend on prior context), return it unchanged.
- Preserve the original intent exactly — do not answer the question, only rewrite it.
- Keep it concise — one question, no preamble, no explanation.
- Output ONLY the rewritten question, nothing else."""

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=CONDENSE_MODEL, temperature=0.0, max_tokens=150)
    return _llm


def format_history(chat_history: list[dict]) -> str:
    """chat_history: list of {'query': ..., 'answer': ...} dicts, oldest first."""
    lines = []
    for turn in chat_history:
        lines.append(f"Developer asked: {turn['query']}")
        lines.append(f"Assistant answered: {turn['answer']}")
    return "\n".join(lines)


def condense_query(query: str, chat_history: list[dict]) -> str:
    """
    Rewrite `query` into a standalone question using `chat_history` as context.
    If there's no history yet (first turn), returns the query unchanged —
    no LLM call needed, and no risk of the model "fixing" something that
    wasn't broken.
    """
    if not chat_history:
        return query

    llm = _get_llm()
    history_text = format_history(chat_history)
    user_content = f"CONVERSATION SO FAR:\n{history_text}\n\nFOLLOW-UP QUESTION:\n{query}"

    messages = [
        SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    rewritten = response.content.strip()
    return rewritten if rewritten else query