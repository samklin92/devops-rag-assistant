"""RAG pipeline: retrieve evidence, generate answers, and optionally evaluate grounding."""

from sentence_transformers import SentenceTransformer

from app.answer_grounding import evaluate_answer_grounding
from app.embedder import create_embedder
from app.generator import create_client, generate_answer
from app.retriever import retrieve


def answer_question(
    question: str,
    embedding_model: SentenceTransformer,
    client,
    top_k: int = 3,
    evaluate_grounding: bool = False,
) -> dict:
    """Generate a grounded answer and optionally evaluate its factual support."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Step 1: Retrieve relevant chunks from the persisted vector index.
    retrieved_chunks = retrieve(
        query=question,
        model=embedding_model,
        top_k=top_k,
    )

    # Step 2: Generate an answer using the retrieved evidence.
    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        client=client,
    )

    # Step 3: Build source metadata directly from retrieved chunks.
    sources = []
    seen_sources = set()

    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown source")
        section = chunk.get("section", "Unknown section")

        # Avoid listing the same source and section more than once.
        source_key = (source, section)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "source": source,
                "section": section,
                "score": chunk.get("score"),
            }
        )

    # Step 4: Evaluate grounding only when explicitly requested.
    grounding_evaluation = None

    if evaluate_grounding and retrieved_chunks:
        grounding_evaluation = evaluate_answer_grounding(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            client=client,
        )

    return {
        "answer": answer,
        "sources": sources,
        "grounding_evaluation": grounding_evaluation,
    }


def main() -> None:
    """Run the complete RAG workflow from the command line."""

    question = input("Enter your DevOps question: ").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    print("\nLoading embedding model...")
    embedding_model = create_embedder()

    print("Connecting to the LLM provider...")
    client = create_client()

    print("Retrieving knowledge and generating an answer...")

    result = answer_question(
        question=question,
        embedding_model=embedding_model,
        client=client,
        top_k=3,
        evaluate_grounding=False,
    )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    if not result["sources"]:
        print("No source documents were retrieved.")
    else:
        for source in result["sources"]:
            score = source["score"]
            score_text = (
                f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            )

            print(f"- Document: {source['source']}")
            print(f"  Section: {source['section']}")
            print(f"  Similarity: {score_text}")

    if result["grounding_evaluation"] is not None:
        evaluation = result["grounding_evaluation"]

        print("\n" + "=" * 60)
        print("ANSWER-GROUNDING EVALUATION")
        print("=" * 60)
        print(f"Grounded: {evaluation['grounded']}")
        print(f"Claims evaluated: {evaluation['claim_count']}")
        print(
            "Supported claims: "
            f"{evaluation['supported_claim_count']}"
        )
        print(
            "Unsupported claims: "
            f"{len(evaluation['unsupported_claims'])}"
        )
        print(
            "Contradicted claims: "
            f"{len(evaluation['contradicted_claims'])}"
        )

        for claim in (
            evaluation["unsupported_claims"]
            + evaluation["contradicted_claims"]
        ):
            print(f"- Claim: {claim['claim']}")
            print(f"  Verdict: {claim['verdict']}")
            print(f"  Reason: {claim['reason']}")


if __name__ == "__main__":
    main()
