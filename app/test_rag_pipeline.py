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

def test_answer_question_skips_grounding_evaluation_by_default(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        rag_pipeline,
        "retrieve",
        lambda query, model, top_k: sample_chunks(),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, retrieved_chunks, client: "Test answer.",
    )

    def unexpected_evaluation(**kwargs):
        pytest.fail("Grounding evaluation should be disabled by default.")

    monkeypatch.setattr(
        rag_pipeline,
        "evaluate_answer_grounding",
        unexpected_evaluation,
    )

    result = rag_pipeline.answer_question(
        question="Why is my pod restarting?",
        embedding_model=FakeModel(),
        client=FakeClient(),
    )

    assert result["answer"] == "Test answer."
    assert result["grounding_evaluation"] is None


def test_answer_question_runs_grounding_evaluation_when_enabled(
    monkeypatch,
):
    chunks = sample_chunks()
    captured = {}

    def fake_retrieve(query, model, top_k):
        return chunks

    def fake_generate_answer(question, retrieved_chunks, client):
        return "Inspect the previous container logs."

    def fake_evaluate_answer_grounding(
        question,
        answer,
        retrieved_chunks,
        client,
    ):
        captured["question"] = question
        captured["answer"] = answer
        captured["retrieved_chunks"] = retrieved_chunks
        captured["client"] = client

        return {
            "grounded": True,
            "claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
            "claim_count": 0,
            "supported_claim_count": 0,
        }

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        fake_generate_answer,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "evaluate_answer_grounding",
        fake_evaluate_answer_grounding,
    )

    model = FakeModel()
    client = FakeClient()
    question = "Why does my pod keep restarting?"

    result = rag_pipeline.answer_question(
        question=question,
        embedding_model=model,
        client=client,
        evaluate_grounding=True,
    )

    assert result["grounding_evaluation"]["grounded"] is True
    assert captured["question"] == question
    assert captured["answer"] == "Inspect the previous container logs."
    assert captured["retrieved_chunks"] == chunks
    assert captured["client"] is client


def test_answer_question_skips_grounding_when_no_chunks_are_retrieved(
    monkeypatch,
):
    monkeypatch.setattr(
        rag_pipeline,
        "retrieve",
        lambda query, model, top_k: [],
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, retrieved_chunks, client: (
            "I could not find relevant information."
        ),
    )

    def unexpected_evaluation(**kwargs):
        pytest.fail("Cannot evaluate grounding without retrieved evidence.")

    monkeypatch.setattr(
        rag_pipeline,
        "evaluate_answer_grounding",
        unexpected_evaluation,
    )

    result = rag_pipeline.answer_question(
        question="How much does a SaaS subscription cost?",
        embedding_model=FakeModel(),
        client=FakeClient(),
        evaluate_grounding=True,
    )

    assert result["grounding_evaluation"] is None
    assert result["sources"] == []
