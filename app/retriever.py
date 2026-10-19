from sentence_transformers import SentenceTransformer

from app.embedder import create_embedder
from app.vector_search import search_chunks
from app.vector_store import load_index


def retrieve(
    query: str,
    model: SentenceTransformer,
    top_k: int = 3,
    min_score: float = 0.40,
) -> list[dict]:
    """Retrieve relevant knowledge chunks for a user's question."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1.0 and 1.0.")

    embedded_chunks = load_index()

    return search_chunks(
        query=query,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=top_k,
        min_score=min_score,
    )


def main() -> None:
    """Run a retrieval example from the command line."""

    query = input("Enter your DevOps question: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    model = create_embedder()

    results = retrieve(
        query=query,
        model=model,
        top_k=3,
        min_score=0.35,
    )

    if not results:
        print("No relevant document chunks were found.")
        return

    print(f"\nQuestion: {query}")
    print(f"Results returned: {len(results)}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Section: {result['section']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Content:\n{result['content']}")


if __name__ == "__main__":
    main()
