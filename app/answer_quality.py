"""Aggregate end-to-end RAG answer-quality evaluation results."""

from typing import Any


ASSESSMENT_FIELDS = (
    "answer_relevant",
    "grounded",
    "answerable",
    "answerability_handled",
)


def _validate_results(results: list[dict[str, Any]]) -> None:
    """Validate the evaluation results and assessment values."""
    if not isinstance(results, list):
        raise TypeError("results must be a list of dictionaries")

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise TypeError(
                f"result at index {index} must be a dictionary"
            )

        for field in ASSESSMENT_FIELDS:
            value = result.get(field)

            if value is not None and not isinstance(value, bool):
                raise TypeError(
                    f"{field} at index {index} must be True, False, or None"
                )


def summarize_answer_quality(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize answer relevance, grounding, and answerability handling."""
    _validate_results(results)

    total_questions = len(results)

    relevance_results = [
        result["answer_relevant"]
        for result in results
        if result.get("answer_relevant") is not None
    ]

    grounding_results = [
        result["grounded"]
        for result in results
        if result.get("grounded") is not None
    ]

    answerability_results = [
        result["answerable"]
        for result in results
        if result.get("answerable") is not None
    ]

    handling_results = [
        result["answerability_handled"]
        for result in results
        if result.get("answerability_handled") is not None
    ]

    return {
        "total_questions": total_questions,
        "relevance_evaluated": len(relevance_results),
        "relevance_rate": (
            sum(relevance_results) / len(relevance_results)
            if relevance_results
            else None
        ),
        "grounding_evaluated": len(grounding_results),
        "grounding_rate": (
            sum(grounding_results) / len(grounding_results)
            if grounding_results
            else None
        ),
        "answerability_evaluated": len(answerability_results),
        "answerability_handling_evaluated": len(handling_results),
        "answerability_handling_rate": (
            sum(handling_results) / len(handling_results)
            if handling_results
            else None
        ),
    }