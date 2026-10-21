from sentence_transformers import CrossEncoder

from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.reranker import rerank_chunks
from app.vector_search import search_chunks


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QUERY = (
    "How can I investigate database authentication failures "
    "without exposing passwords?"
)

EXPECTED_SOURCE = "database-connectivity.md"


def print_results(title: str, results: list[dict]) -> None:
    print(f"\n{title}")

    if not results:
        print("No results.")
        return

    for rank, chunk in enumerate(results, start=1):
        print(
            f"{rank}. {chunk['source']} | "
            f"{chunk['section']} | "
            f"similarity={chunk['score']:.4f}"
            + (
                f" | rerank_score={chunk['rerank_score']:.4f}"
                if "rerank_score" in chunk
                else ""
            )
        )


def contains_expected_source(results: list[dict]) -> bool:
    return any(
        chunk["source"] == EXPECTED_SOURCE
        for chunk in results
    )


def main() -> None:
    print("Loading knowledge base...")

    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents were loaded.")

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    if not chunks:
        raise RuntimeError("No document chunks were created.")

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")
    embedding_model = create_embedder()

    print("Embedding document chunks...")
    embedded_chunks = embed_chunks(chunks, embedding_model)

    print("Loading cross-encoder reranker...")
    reranker_model = CrossEncoder(RERANKER_MODEL)

    # Stage 1: Retrieve a broad set of candidate chunks.
    candidates = search_chunks(
        query=QUERY,
        embedded_chunks=embedded_chunks,
        model=embedding_model,
        top_k=10,
        min_score=-1.0,
    )

    if not candidates:
        raise RuntimeError("Vector search returned no candidates.")

    # Compare the original top three with the reranked top three.
    original_top_three = candidates[:3]

    reranked_top_three = rerank_chunks(
        query=QUERY,
        chunks=candidates,
        model=reranker_model,
        top_k=3,
    )

    print_results("Original vector-search top 3", original_top_three)
    print_results("Cross-encoder reranked top 3", reranked_top_three)

    print("\nEvaluation")
    print(
        "Expected document in original top 3: "
        f"{contains_expected_source(original_top_three)}"
    )
    print(
        "Expected document in reranked top 3: "
        f"{contains_expected_source(reranked_top_three)}"
    )


if __name__ == "__main__":
    main()
