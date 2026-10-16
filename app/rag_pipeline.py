from sentence_transformers import SentenceTransformer

from app.embedder import create_embedder
from app.generator import create_client, generate_answer
from app.retriever import retrieve


def answer_question(
    question: str,
    embedding_model: SentenceTransformer,
    client,
    top_k: int = 3,
) -> str:
    """Retrieve relevant knowledge and generate an evidence-based answer."""

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

    return answer


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

    answer = answer_question(
        question=question,
        embedding_model=embedding_model,
        client=client,
        top_k=3,
    )

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()