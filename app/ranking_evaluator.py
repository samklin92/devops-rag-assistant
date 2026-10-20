import re
from typing import Any


def normalize_section(section: str) -> str:
    """Normalize Markdown heading markers and whitespace for comparison."""
    section = re.sub(r"^\s*#{1,6}\s*", "", section)
    return " ".join(section.split())


def reciprocal_rank(
    retrieved_chunks: list[dict[str, Any]],
    relevant_sections: list[dict[str, str]],
) -> float:
    """Return the reciprocal rank of the first relevant retrieved chunk."""
    if not relevant_sections:
        raise ValueError("relevant_sections cannot be empty.")

    relevant = {
        (item["source"], normalize_section(item["section"]))
        for item in relevant_sections
    }

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        key = (chunk["source"], normalize_section(chunk["section"]))
        if key in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    question_results: list[dict[str, Any]],
) -> float:
    """Calculate mean reciprocal rank across evaluated questions."""
    if not question_results:
        raise ValueError("question_results cannot be empty.")

    reciprocal_ranks = []
    for result in question_results:
        if "retrieved_chunks" not in result:
            raise ValueError(
                f"Question {result.get('id', '<unknown>')} is missing "
                "'retrieved_chunks'."
            )
        if "relevant_sections" not in result:
            raise ValueError(
                f"Question {result.get('id', '<unknown>')} is missing "
                "'relevant_sections'."
            )
        reciprocal_ranks.append(
            reciprocal_rank(
                result["retrieved_chunks"], result["relevant_sections"]
            )
        )

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
