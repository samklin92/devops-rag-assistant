import pytest

from app import retriever


def sample_chunks():
    return [
        {
            "source": "crashloopbackoff.md",
            "section": "Symptoms",
            "content": "The pod repeatedly restarts.",
            "embedding": [0.1, 0.2],
        },
        {
            "source": "oomkilled.md",
            "section": "Memory",
            "content": "The container exceeded its memory limit.",
            "embedding": [0.3, 0.4],
        },
    ]


class FakeModel:
    """A placeholder model for testing retrieval orchestration."""


def test_retrieve_returns_search_results(monkeypatch):
    chunks = sample_chunks()
    expected_results = [
        {
            "source": "crashloopbackoff.md",
            "section": "Symptoms",
            "content": "The pod repeatedly restarts.",
            "score": 0.9,
        }
    ]

    monkeypatch.setattr(
        retriever,
        "load_index",
        lambda: chunks,
    )

    def fake_search(query, embedded_chunks, model, top_k):
        assert query == "Why is my pod restarting?"
        assert embedded_chunks == chunks
        assert top_k == 3
        return expected_results

    monkeypatch.setattr(
        retriever,
        "search_chunks",
        fake_search,
    )

    results = retriever.retrieve(
        query="Why is my pod restarting?",
        model=FakeModel(),
    )

    assert results == expected_results


def test_retrieve_passes_top_k_to_search(monkeypatch):
    chunks = sample_chunks()
    captured = {}

    monkeypatch.setattr(
        retriever,
        "load_index",
        lambda: chunks,
    )

    def fake_search(query, embedded_chunks, model, top_k):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(
        retriever,
        "search_chunks",
        fake_search,
    )

    retriever.retrieve(
        query="How do I investigate a pod?",
        model=FakeModel(),
        top_k=2,
    )

    assert captured["top_k"] == 2


def test_retrieve_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve(
            query="   ",
            model=FakeModel(),
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_invalid_top_k(top_k):
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            query="Why is my pod restarting?",
            model=FakeModel(),
            top_k=top_k,
        )


def test_retrieve_returns_empty_for_empty_index(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "load_index",
        lambda: [],
    )

    def fake_search(query, embedded_chunks, model, top_k):
        return []

    monkeypatch.setattr(
        retriever,
        "search_chunks",
        fake_search,
    )

    results = retriever.retrieve(
        query="Why is my pod restarting?",
        model=FakeModel(),
    )

    assert results == []