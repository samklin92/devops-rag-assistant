import pytest

from app.ranking_evaluator import (
    mean_reciprocal_rank,
    reciprocal_rank,
)


def make_chunk(source: str, section: str) -> dict:
    return {"source": source, "section": section, "content": "Example"}


def test_reciprocal_rank_returns_one_for_first_result():
    chunks = [
        make_chunk("crashloopbackoff.md", "Symptoms"),
        make_chunk("oomkilled.md", "Overview"),
    ]
    relevant = [
        {"source": "crashloopbackoff.md", "section": "Symptoms"}
    ]

    assert reciprocal_rank(chunks, relevant) == 1.0


def test_reciprocal_rank_returns_half_for_second_result():
    chunks = [
        make_chunk("oomkilled.md", "Overview"),
        make_chunk("crashloopbackoff.md", "Symptoms"),
    ]
    relevant = [
        {"source": "crashloopbackoff.md", "section": "Symptoms"}
    ]

    assert reciprocal_rank(chunks, relevant) == 0.5


def test_reciprocal_rank_returns_zero_when_not_found():
    chunks = [make_chunk("oomkilled.md", "Overview")]
    relevant = [
        {"source": "crashloopbackoff.md", "section": "Symptoms"}
    ]

    assert reciprocal_rank(chunks, relevant) == 0.0


def test_reciprocal_rank_accepts_any_relevant_section():
    chunks = [
        make_chunk("oomkilled.md", "Overview"),
        make_chunk("database-connectivity.md", "DNS"),
    ]
    relevant = [
        {"source": "database-connectivity.md", "section": "DNS"},
        {"source": "crashloopbackoff.md", "section": "Symptoms"},
    ]

    assert reciprocal_rank(chunks, relevant) == 0.5


def test_reciprocal_rank_rejects_empty_relevance_labels():
    with pytest.raises(ValueError, match="cannot be empty"):
        reciprocal_rank([], [])


def test_mean_reciprocal_rank_averages_results():
    results = [
        {
            "id": "q01",
            "retrieved_chunks": [
                make_chunk("crashloopbackoff.md", "Symptoms")
            ],
            "relevant_sections": [
                {"source": "crashloopbackoff.md", "section": "Symptoms"}
            ],
        },
        {
            "id": "q02",
            "retrieved_chunks": [
                make_chunk("oomkilled.md", "Overview"),
                make_chunk("oomkilled.md", "Termination"),
            ],
            "relevant_sections": [
                {"source": "oomkilled.md", "section": "Termination"}
            ],
        },
    ]

    assert mean_reciprocal_rank(results) == 0.75


def test_mean_reciprocal_rank_rejects_empty_results():
    with pytest.raises(ValueError, match="cannot be empty"):
        mean_reciprocal_rank([])


def test_mean_reciprocal_rank_rejects_missing_fields():
    with pytest.raises(ValueError, match="retrieved_chunks"):
        mean_reciprocal_rank([{"id": "q01"}])
