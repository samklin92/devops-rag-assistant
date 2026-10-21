"""Evaluate retrieval behavior against expected answerability labels."""

from typing import Any


def evaluate_answerability_retrieval(
    question_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize retrieval outcomes for answerable and unanswerable questions."""

    if not isinstance(question_results, list):
        raise TypeError("question_results must be a list")

    for index, result in enumerate(question_results):
        if not isinstance(result, dict):
            raise TypeError(f"Result at index {index} must be a dictionary")

        if not isinstance(result.get("answerable"), bool):
            raise ValueError(
                f"Result at index {index} must contain a boolean 'answerable'"
            )

        if not isinstance(result.get("expected_documents"), list):
            raise ValueError(
                f"Result at index {index} must contain an "
                "'expected_documents' list"
            )

        if not isinstance(result.get("retrieved_documents"), list):
            raise ValueError(
                f"Result at index {index} must contain a "
                "'retrieved_documents' list"
            )

    answerable_results = [
        result
        for result in question_results
        if result["answerable"] is True
    ]

    unanswerable_results = [
        result
        for result in question_results
        if result["answerable"] is False
    ]

    successful_retrievals = 0

    for result in answerable_results:
        expected = set(result["expected_documents"])
        retrieved = set(result["retrieved_documents"])

        if expected.intersection(retrieved):
            successful_retrievals += 1

    unanswerable_retrievals = sum(
        bool(result["retrieved_documents"])
        for result in unanswerable_results
    )

    return {
        "total_questions": len(question_results),
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": len(unanswerable_results),
        "answerable_retrieval_success_rate": (
            successful_retrievals / len(answerable_results)
            if answerable_results
            else None
        ),
        "unanswerable_document_retrieval_rate": (
            unanswerable_retrievals / len(unanswerable_results)
            if unanswerable_results
            else None
        ),
    }