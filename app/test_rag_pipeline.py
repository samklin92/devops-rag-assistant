"""Unit tests for app.rag_pipeline.answer_question."""

import pytest

from app import rag_pipeline


class FakeModel:
    """Placeholder for the embedding model during unit tests."""


class FakeClient:
    """Placeholder for the LLM client during unit tests."""


def sample_chunks():
    """Return one retrieved chunk in the format retrieval produces."""
    return [
        {
            "source": "crashloopbackoff.md",
            "section": "Diagnostic Commands",
            "content": "Use kubectl logs and kubectl describe pod.",
            "score": 0.85,
        }
    ]


def test_answer_question_connects_retrieval_and_generation(monkeypatch):
    chunks = sample_chunks()
    captured = {}

    def fake_retrieve(query, model, top_k):
        captured["query"] = query
        captured["model"] = model
        captured["top_k"] = top_k
        return chunks

    def fake_generate_answer(question, retrieved_chunks, client):
        captured["question"] = question
        captured["retrieved_chunks"] = retrieved_chunks
        captured["client"] = client
        return "Inspect the previous container logs."

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        fake_generate_answer,
    )

    model = FakeModel()
    client = FakeClient()
    question = "Why does my Kubernetes pod keep restarting?"

    result = rag_pipeline.answer_question(
        question=question,
        embedding_model=model,
        client=client,
        top_k=3,
    )

    assert result["answer"] == "Inspect the previous container logs."
    assert result["sources"] == [
        {
            "source": "crashloopbackoff.md",
            "section": "Diagnostic Commands",
            "score": 0.85,
        }
    ]
    assert captured["query"] == question
    assert captured["question"] == question
    assert captured["model"] is model
    assert captured["client"] is client
    assert captured["top_k"] == 3
    assert captured["retrieved_chunks"] == chunks


def test_answer_question_passes_custom_top_k(monkeypatch):
    captured = {}

    def fake_retrieve(query, model, top_k):
        captured["top_k"] = top_k
        return sample_chunks()

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, retrieved_chunks, client: "Test answer.",
    )

    rag_pipeline.answer_question(
        question="How do I investigate a pod?",
        embedding_model=FakeModel(),
        client=FakeClient(),
        top_k=2,
    )

    assert captured["top_k"] == 2


def test_answer_question_rejects_empty_question():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        rag_pipeline.answer_question(
            question=" ",
            embedding_model=FakeModel(),
            client=FakeClient(),
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_answer_question_rejects_invalid_top_k(top_k):
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        rag_pipeline.answer_question(
            question="Why is my pod restarting?",
            embedding_model=FakeModel(),
            client=FakeClient(),
            top_k=top_k,
        )