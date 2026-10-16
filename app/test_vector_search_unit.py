import numpy as np
import pytest

from app.vector_search import search_chunks


class FakeModel:
    """Return a controlled query embedding for testing."""

    def encode(self, text, **kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)


def make_chunk(source, section, content, embedding):
    return {
        "source": source,
        "section": section,
        "content": content,
        "embedding": np.array(embedding, dtype=np.float32),
    }


def test_search_ranks_most_similar_chunk_first():
    chunks = [
        make_chunk(
            "database.md",
            "Database",
            "Database connection troubleshooting",
            [0.0, 1.0],
        ),
        make_chunk(
            "crashloopbackoff.md",
            "CrashLoopBackOff",
            "Troubleshooting restarting Kubernetes pods",
            [1.0, 0.0],
        ),
    ]

    results = search_chunks(
        query="Why does my pod keep restarting?",
        embedded_chunks=chunks,
        model=FakeModel(),
        top_k=2,
    )

    assert results[0]["source"] == "crashloopbackoff.md"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_top_k_limits_results():
    chunks = [
        make_chunk("a.md", "A", "First", [1.0, 0.0]),
        make_chunk("b.md", "B", "Second", [0.0, 1.0]),
        make_chunk("c.md", "C", "Third", [-1.0, 0.0]),
    ]

    results = search_chunks(
        query="test query",
        embedded_chunks=chunks,
        model=FakeModel(),
        top_k=2,
    )

    assert len(results) == 2


def test_empty_chunks_return_empty_list():
    results = search_chunks(
        query="test query",
        embedded_chunks=[],
        model=FakeModel(),
    )

    assert results == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        search_chunks(
            query="   ",
            embedded_chunks=[],
            model=FakeModel(),
        )


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        search_chunks(
            query="test query",
            embedded_chunks=[],
            model=FakeModel(),
            top_k=0,
        )


def test_metadata_is_preserved():
    chunks = [
        make_chunk(
            "crashloopbackoff.md",
            "Symptoms",
            "The restart count keeps increasing.",
            [1.0, 0.0],
        )
    ]

    results = search_chunks(
        query="test query",
        embedded_chunks=chunks,
        model=FakeModel(),
    )

    assert results[0]["source"] == "crashloopbackoff.md"
    assert results[0]["section"] == "Symptoms"
    assert results[0]["content"] == "The restart count keeps increasing."
