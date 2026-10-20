import numpy as np
import pytest

from app.reranker import rerank_chunks


class FakeCrossEncoder:
    """Return predefined scores without downloading a model."""

    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)
        self.received_pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.received_pairs = pairs
        return self.scores


@pytest.fixture
def sample_chunks():
    return [
        {
            "source": "database-connectivity.md",
            "section": "Inspect the Application Pod",
            "content": "Check pod status and inspect application logs.",
            "score": 0.5088,
        },
        {
            "source": "crashloopbackoff.md",
            "section": "Troubleshooting Steps",
            "content": "Investigate failed liveness probes and container restarts.",
            "score": 0.4926,
        },
        {
            "source": "oomkilled.md",
            "section": "Verification",
            "content": "Check container memory usage and OOMKilled events.",
            "score": 0.4805,
        },
    ]


def test_rerank_chunks_sorts_by_rerank_score(sample_chunks):
    model = FakeCrossEncoder([0.30, 0.95, 0.50])

    results = rerank_chunks(
        query="Why is my container restarting because of a failed liveness probe?",
        chunks=sample_chunks,
        model=model,
        top_k=3,
    )

    assert [chunk["source"] for chunk in results] == [
        "crashloopbackoff.md",
        "oomkilled.md",
        "database-connectivity.md",
    ]

    assert [chunk["rerank_score"] for chunk in results] == [
        0.95,
        0.50,
        0.30,
    ]


def test_rerank_chunks_preserves_original_metadata(sample_chunks):
    model = FakeCrossEncoder([0.30, 0.95, 0.50])

    results = rerank_chunks(
        query="Why is my container restarting?",
        chunks=sample_chunks,
        model=model,
        top_k=3,
    )

    best_chunk = results[0]

    assert best_chunk["section"] == "Troubleshooting Steps"
    assert best_chunk["content"] == sample_chunks[1]["content"]
    assert best_chunk["score"] == 0.4926
    assert best_chunk["rerank_score"] == 0.95


def test_rerank_chunks_limits_results(sample_chunks):
    model = FakeCrossEncoder([0.30, 0.95, 0.50])

    results = rerank_chunks(
        query="Why is my container restarting?",
        chunks=sample_chunks,
        model=model,
        top_k=2,
    )

    assert len(results) == 2


def test_rerank_chunks_returns_empty_list_for_no_chunks():
    model = FakeCrossEncoder([])

    results = rerank_chunks(
        query="Why is my container restarting?",
        chunks=[],
        model=model,
    )

    assert results == []


def test_rerank_chunks_rejects_empty_query(sample_chunks):
    model = FakeCrossEncoder([0.30, 0.95, 0.50])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        rerank_chunks(
            query="   ",
            chunks=sample_chunks,
            model=model,
        )


def test_rerank_chunks_rejects_invalid_top_k(sample_chunks):
    model = FakeCrossEncoder([0.30, 0.95, 0.50])

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        rerank_chunks(
            query="Why is my container restarting?",
            chunks=sample_chunks,
            model=model,
            top_k=0,
        )


def test_rerank_chunks_passes_query_and_content_to_model(sample_chunks):
    model = FakeCrossEncoder([0.30, 0.95, 0.50])
    query = "Why is my container restarting?"

    rerank_chunks(
        query=query,
        chunks=sample_chunks,
        model=model,
    )

    assert model.received_pairs == [
        (query, chunk["content"])
        for chunk in sample_chunks
    ]
