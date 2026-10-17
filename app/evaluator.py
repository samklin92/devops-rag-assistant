import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "evals" / "questions.json"
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base" / "documents"


def load_evaluation_questions() -> list[dict]:
    """Load questions and expected documents from the evaluation dataset."""

    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must contain a list of questions.")

    return questions


def evaluate_retrieval(
    question: dict,
    retrieved_chunks: list[dict],
) -> dict:
    """Evaluate document-level retrieval recall for one question."""

    expected_documents = set(question["expected_documents"])
    answerable = question["answerable"]

    # Preserve retrieval order while removing duplicate document sources.
    retrieved_documents = list(
        dict.fromkeys(
            chunk["source"]
            for chunk in retrieved_chunks
        )
    )

    # Identify expected documents that are not present in the knowledge base.
    existing_documents = {
        path.name
        for path in KNOWLEDGE_BASE.glob("*.md")
    }

    missing_documents = sorted(
        expected_documents - existing_documents
    )

    # Recall is meaningful only for answerable questions with expected documents.
    if not answerable:
        recall = None
        matched_documents = []
    else:
        matched_documents = sorted(
            expected_documents.intersection(retrieved_documents)
        )

        recall = (
            len(matched_documents) / len(expected_documents)
            if expected_documents
            else 0.0
        )

    return {
        "id": question["id"],
        "question": question["question"],
        "answerable": answerable,
        "expected_documents": sorted(expected_documents),
        "retrieved_documents": retrieved_documents,
        "matched_documents": matched_documents,
        "missing_documents": missing_documents,
        "recall": recall,
    }


def run_evaluation(
    top_k: int = 3,
    min_score: float = 0.35,
) -> list[dict]:
    """Evaluate retrieval against the complete question dataset."""

    from app.embedder import create_embedder
    from app.retriever import retrieve

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    questions = load_evaluation_questions()
    model = create_embedder()

    results = []

    for question in questions:
        chunks = retrieve(
            query=question["question"],
            model=model,
            top_k=top_k,
            min_score=min_score,
        )

        result = evaluate_retrieval(
            question=question,
            retrieved_chunks=chunks,
        )

        results.append(result)

        print(f"\n[{result['id']}] {result['question']}")
        print(f"Answerable: {result['answerable']}")
        print(f"Expected: {result['expected_documents']}")
        print(f"Retrieved: {result['retrieved_documents']}")

        if result["recall"] is not None:
            print(f"Recall@{top_k}: {result['recall']:.2f}")
        else:
            print("Recall: N/A (unanswerable question)")

        if result["missing_documents"]:
            print(
                "Missing from knowledge base: "
                f"{result['missing_documents']}"
            )

    # Calculate mean Recall@K for answerable questions.
    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    if answerable_results:
        mean_recall = sum(
            result["recall"]
            for result in answerable_results
        ) / len(answerable_results)

        print(f"\nMean Recall@{top_k}: {mean_recall:.2f}")
    else:
        print("\nMean Recall: N/A (no answerable questions)")

    # Measure how often unanswerable questions still retrieve documents.
    unanswerable_results = [
        result
        for result in results
        if not result["answerable"]
    ]

    if unanswerable_results:
        false_retrievals = sum(
            bool(result["retrieved_documents"])
            for result in unanswerable_results
        )

        unanswerable_retrieval_rate = (
            false_retrievals / len(unanswerable_results)
        )

        print(
            "Unanswerable retrieval rate: "
            f"{unanswerable_retrieval_rate:.2f}"
        )
    else:
        print("Unanswerable retrieval rate: N/A")

    return results


if __name__ == "__main__":
    run_evaluation()
