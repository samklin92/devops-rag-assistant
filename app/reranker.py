from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_chunks(
    query: str,
    chunks: list[dict],
    model: CrossEncoder,
    top_k: int = 3,
) -> list[dict]:
    """Rerank retrieved chunks by their relevance to a query."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not chunks:
        return []

    # Prepare query-and-document pairs for the cross-encoder.
    pairs = [
        (query, chunk["content"])
        for chunk in chunks
    ]

    # Score each candidate against the query.
    scores = model.predict(pairs, show_progress_bar=False)

    # Preserve the original chunk metadata while adding the reranking score.
    reranked_chunks = []

    for chunk, score in zip(chunks, scores):
        reranked_chunk = chunk.copy()
        reranked_chunk["rerank_score"] = float(score)
        reranked_chunks.append(reranked_chunk)

    # Sort by reranking score, highest first.
    reranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return reranked_chunks[:top_k]
