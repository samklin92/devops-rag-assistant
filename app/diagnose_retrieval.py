from app.embedder import create_embedder
from app.vector_store import load_index
from app.vector_search import search_chunks

QUESTIONS = [
    {
        "id": "q16",
        "question": "My Kubernetes pod shows ImagePullBackOff. How do I diagnose and fix the image pull failure?",
    }
]


def main():
    model = create_embedder()
    chunks = load_index()

    print(f"Indexed chunks: {len(chunks)}")

    for item in QUESTIONS:
        print(f"\n{'=' * 70}")
        print(f"{item['id']}: {item['question']}")
        print("=" * 70)

        results = search_chunks(
            query=item["question"],
            embedded_chunks=chunks,
            model=model,
            top_k=5,
            min_score=-1.0,
        )

        if not results:
            print("No chunks found in the index.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Source: {result['source']}")
            print(f"Section: {result['section']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Content: {result['content'][:350]}")


if __name__ == "__main__":
    main()
