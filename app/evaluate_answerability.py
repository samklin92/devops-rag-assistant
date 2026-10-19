"""Evaluate retrieval answerability against the project's test questions."""

import json
from pathlib import Path

from app.answerability_evaluator import evaluate_answerability_retrieval
from app.embedder import create_embedder
from app.retriever import retrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "evals" / "questions.json"


def load_questions() -> list[dict]:
    """Load and validate the evaluation questions."""
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("Evaluation questions must be a JSON list.")

    required_fields = {
        "id",
        "question",
        "expected_documents",
        "answerable",
    }

    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(
                f"Question at index {index} must be a JSON object."
            )

        missing_fields = required_fields - question.keys()

        if missing_fields:
            raise ValueError(
                f"Question at index {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not isinstance(question["answerable"], bool):
            raise ValueError(
                f"Question {question['id']} must have a boolean answerable field."
            )

        if not isinstance(question["expected_documents"], list):
            raise ValueError(
                f"Question {question['id']} must have an expected_documents list."
            )

    return questions


def run_evaluation(min_scores: tuple[float, ...] = (0.50,)) -> None:
    """Run retrieval evaluation and print individual and aggregate results."""
    questions = load_questions()

    if not questions:
        print("No evaluation questions were found.")
        return

    print(f"Loaded {len(questions)} evaluation questions.")
    print("Loading embedding model...")

    model = create_embedder()

    for min_score in min_scores:
        results = []

        print("\n" + "=" * 60)
        print(f"EVALUATION WITH MINIMUM SCORE: {min_score:.2f}")
        print("=" * 60)

        for question in questions:
            retrieved_chunks = retrieve(
                query=question["question"],
                model=model,
                top_k=3,
                min_score=min_score,
            )

            retrieved_documents = sorted(
                {chunk["source"] for chunk in retrieved_chunks}
            )

            result = {
                "id": question["id"],
                "question": question["question"],
                "answerable": question["answerable"],
                "expected_documents": question["expected_documents"],
                "retrieved_documents": retrieved_documents,
            }

            results.append(result)

            expected = set(question["expected_documents"])
            retrieved = set(retrieved_documents)

            if question["answerable"]:
                success = bool(expected.intersection(retrieved))
                status = "PASS" if success else "MISS"
            else:
                success = not retrieved
                status = "PASS" if success else "REVIEW"

            print(f"\n[{status}] {question['id']}: {question['question']}")
            print(f"  Expected documents: {sorted(expected) or 'None'}")
            print(f"  Retrieved documents: {retrieved_documents or 'None'}")

        summary = evaluate_answerability_retrieval(results)

        print("\n" + "-" * 60)
        print(f"SUMMARY FOR THRESHOLD {min_score:.2f}")
        print("-" * 60)
        print(f"Total questions: {summary['total_questions']}")
        print(f"Answerable questions: {summary['answerable_questions']}")
        print(f"Unanswerable questions: {summary['unanswerable_questions']}")

        success_rate = summary["answerable_retrieval_success_rate"]
        unanswerable_rate = summary["unanswerable_retrieval_rate"]

        if success_rate is not None:
            print(f"Answerable retrieval success rate: {success_rate:.1%}")

        if unanswerable_rate is not None:
            print(f"Unanswerable retrieval rate: {unanswerable_rate:.1%}")

if __name__ == "__main__":
        run_evaluation(min_scores=(0.50, 0.45, 0.40))