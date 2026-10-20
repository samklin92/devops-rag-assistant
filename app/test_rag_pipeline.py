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

def test_answer_question_uses_vector_retrieval_by_default(monkeypatch):
    captured = {}

    def fake_retrieve(query, model, top_k):
        captured["top_k"] = top_k
        return sample_chunks()

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        rag_pipeline,
        "generate_answer",
        lambda question, retrieved_chunks, client: "Vector-only answer.",
    )

    result = rag_pipeline.answer_question(
        question="Why is my pod restarting?",
        embedding_model=FakeModel(),
        client=FakeClient(),
        top_k=2,
    )

    assert captured["top_k"] == 2
    assert result["answer"] == "Vector-only answer."
    assert "rerank_score" not in result["sources"][0]


def test_answer_question_reranks_candidates_before_generation(monkeypatch):
    chunks = [
        {
            "source": "first.md",
            "section": "First",
            "content": "First candidate.",
            "score": 0.90,
        },
        {
            "source": "second.md",
            "section": "Second",
            "content": "Second candidate.",
            "score": 0.80,
        },
        {
            "source": "third.md",
            "section": "Third",
            "content": "Third candidate.",
            "score": 0.70,
        },
    ]
    captured = {}
    reranker_model = object()

    def fake_retrieve(query, model, top_k):
        captured["candidate_k"] = top_k
        return chunks

    def fake_rerank_chunks(query, chunks, model, top_k):
        captured["reranker_query"] = query
        captured["reranker_model"] = model
        captured["reranker_top_k"] = top_k
        return [
            {**chunks[1], "rerank_score": 0.95},
            {**chunks[0], "rerank_score": 0.40},
        ][:top_k]

    def fake_generate_answer(question, retrieved_chunks, client):
        captured["generated_chunks"] = retrieved_chunks
        return "Reranked answer."

    monkeypatch.setattr(rag_pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_pipeline, "rerank_chunks", fake_rerank_chunks)
    monkeypatch.setattr(rag_pipeline, "generate_answer", fake_generate_answer)

    question = "How do I investigate this issue?"
    result = rag_pipeline.answer_question(
        question=question,
        embedding_model=FakeModel(),
        client=FakeClient(),
        top_k=2,
        use_reranker=True,
        candidate_k=3,
        reranker_model=reranker_model,
    )

    assert captured["candidate_k"] == 3
    assert captured["reranker_query"] == question
    assert captured["reranker_model"] is reranker_model
    assert captured["reranker_top_k"] == 2
    assert [chunk["source"] for chunk in captured["generated_chunks"]] == [
        "second.md",
        "first.md",
    ]
    assert result["sources"][0]["rerank_score"] == 0.95


def test_answer_question_requires_model_when_reranking_enabled():
    with pytest.raises(
        ValueError,
        match="reranker_model must be supplied",
    ):
        rag_pipeline.answer_question(
            question="How do I investigate this issue?",
            embedding_model=FakeModel(),
            client=FakeClient(),
            use_reranker=True,
        )


@pytest.mark.parametrize(
    ("top_k", "candidate_k", "message"),
    [
        (3, 0, "candidate_k must be greater than zero"),
        (3, 2, "candidate_k must be greater than or equal to top_k"),
    ],
)
def test_answer_question_rejects_invalid_reranking_configuration(
    top_k,
    candidate_k,
    message,
):
    with pytest.raises(ValueError, match=message):
        rag_pipeline.answer_question(
            question="How do I investigate this issue?",
            embedding_model=FakeModel(),
            client=FakeClient(),
            top_k=top_k,
            use_reranker=True,
            candidate_k=candidate_k,
            reranker_model=object(),
        )
