import numpy as np
from sentence_transformers import SentenceTransformer


def search_chunks(
    query: str,
    embedded_chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve the most relevant document chunks for a query."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not embedded_chunks:
        return []

    # Convert the user's question into an embedding.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Combine document embeddings into a matrix.
    document_embeddings = np.vstack(
        [chunk["embedding"] for chunk in embedded_chunks]
    )

    # Calculate cosine similarity for normalized vectors.
    scores = document_embeddings @ query_embedding

    # Rank results from highest similarity to lowest.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        chunk = embedded_chunks[int(index)]

        results.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
                "score": float(scores[index]),
            }
        )

    return results