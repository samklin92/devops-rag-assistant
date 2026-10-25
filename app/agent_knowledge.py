"""Retrieve relevant DevOps knowledge for agent investigations."""

from typing import Any

from app.retriever import retrieve


def retrieve_agent_knowledge(
    question: str,
    embedding_model: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve relevant knowledge-base chunks for an agent question."""

    return retrieve(
        query=question,
        model=embedding_model,
        top_k=top_k,
    )