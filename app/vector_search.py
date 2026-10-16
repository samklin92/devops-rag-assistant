import numpy as np
from sentence_transformers import SentenceTransformer


def search_chunks(
    query: str,
    embedded_chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 3,
    min_score: float = 0.50,
) -> list[dict]:
    """Retrieve relevant document chunks using cosine similarity."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1.0 and 1.0.")

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

    # Rank all chunks from highest similarity to lowest.
    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Exclude chunks that fall below the experimental threshold.
        if score < min_score:
            continue

        chunk = embedded_chunks[int(index)]

        results.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
                "score": score,
            }
        )

        # Stop after collecting the requested number of relevant chunks.
        if len(results) >= top_k:
            break

    return results
