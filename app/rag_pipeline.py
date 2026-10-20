"""RAG pipeline: retrieve evidence, generate answers, and optionally evaluate grounding."""

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.answer_grounding import evaluate_answer_grounding
from app.embedder import create_embedder
from app.generator import create_client, generate_answer
from app.reranker import DEFAULT_RERANKER_MODEL, rerank_chunks
from app.retriever import retrieve


def answer_question(
    question: str,
    embedding_model: SentenceTransformer,
    client,
    top_k: int = 3,
    evaluate_grounding: bool = False,
    use_reranker: bool = False,
    candidate_k: int = 10,
    reranker_model: CrossEncoder | None = None,
) -> dict:
    """Generate a grounded answer and optionally evaluate its factual support."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if use_reranker:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        if reranker_model is None:
            raise ValueError(
                "reranker_model must be supplied when use_reranker=True."
            )

    # Step 1: Retrieve candidate chunks using vector similarity.
    retrieval_k = candidate_k if use_reranker else top_k

    retrieved_chunks = retrieve(
        query=question,
        model=embedding_model,
        top_k=retrieval_k,
    )

    # Step 2: Optionally rerank candidates before generating the answer.
    if use_reranker and retrieved_chunks:
        retrieved_chunks = rerank_chunks(
            query=question,
            chunks=retrieved_chunks,
            model=reranker_model,
            top_k=top_k,
        )

    # Step 3: Generate an answer using the selected evidence.
    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        client=client,
    )

    # Step 4: Build source metadata directly from the selected chunks.
    sources = []
    seen_sources = set()

    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown source")
        section = chunk.get("section", "Unknown section")
        source_key = (source, section)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "source": source,
                "section": section,
                "score": chunk.get("score"),
                **(
                    {"rerank_score": chunk["rerank_score"]}
                    if "rerank_score" in chunk
                    else {}
                ),
            }
        )

    # Step 5: Evaluate grounding only when explicitly requested.
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

    # Optional configuration: enable reranking by setting this to True.
    use_reranking = False
    reranker_model = None

    if use_reranking:
        print(f"Loading reranker: {DEFAULT_RERANKER_MODEL}")
        reranker_model = CrossEncoder(DEFAULT_RERANKER_MODEL)

    print("Retrieving knowledge and generating an answer...")

    result = answer_question(
        question=question,
        embedding_model=embedding_model,
        client=client,
        top_k=3,
        evaluate_grounding=False,
        use_reranker=use_reranking,
        candidate_k=10,
        reranker_model=reranker_model,
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
            print(f"  Vector similarity: {score_text}")

            if "rerank_score" in source:
                print(f"  Reranker score: {source['rerank_score']:.4f}")

    if result["grounding_evaluation"] is not None:
        evaluation = result["grounding_evaluation"]

        print("\n" + "=" * 60)
        print("ANSWER-GROUNDING EVALUATION")
        print("=" * 60)
        print(f"Grounded: {evaluation['grounded']}")
        print(f"Claims evaluated: {evaluation['claim_count']}")
        print(f"Supported claims: {evaluation['supported_claim_count']}")
        print(f"Unsupported claims: {len(evaluation['unsupported_claims'])}")
        print(f"Contradicted claims: {len(evaluation['contradicted_claims'])}")

        for claim in (
            evaluation["unsupported_claims"]
            + evaluation["contradicted_claims"]
        ):
            print(f"- Claim: {claim['claim']}")
            print(f"  Verdict: {claim['verdict']}")
            print(f"  Reason: {claim['reason']}")


if __name__ == "__main__":
    main()
