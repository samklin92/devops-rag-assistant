from app.embedder import create_embedder
from app.vector_store import load_index
from app.vector_search import search_chunks

TEST_CASES = [
    {
        "id": "q07",
        "question": (
            "My container keeps restarting because its liveness "
            "probe fails. What should I investigate?"
        ),
        "expected": "crashloopbackoff.md",
    },
    {
        "id": "q08",
        "question": (
            "What should I check if a Kubernetes application "
            "references a missing ConfigMap or Secret?"
        ),
        "expected": "crashloopbackoff.md",
    },
    {
        "id": "q11",
        "question": (
            "How can I investigate database authentication "
            "failures without exposing passwords?"
        ),
        "expected": "database-connectivity.md",
    },
]

THRESHOLDS = [0.50, 0.45, 0.40, 0.35]


def main():
    model = create_embedder()
    chunks = load_index()

    for case in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"{case['id']}: {case['question']}")
        print(f"Expected document: {case['expected']}")
        print("=" * 70)

        for threshold in THRESHOLDS:
            results = search_chunks(
                query=case["question"],
                embedded_chunks=chunks,
                model=model,
                top_k=3,
                min_score=threshold,
            )

            sources = list(dict.fromkeys(
                result["source"] for result in results
            ))

            found = case["expected"] in sources

            print(
                f"Threshold={threshold:.2f} | "
                f"Chunks={len(results)} | "
                f"Sources={sources} | "
                f"Expected document found={found}"
            )


if __name__ == "__main__":
    main()
