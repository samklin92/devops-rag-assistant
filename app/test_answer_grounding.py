import json

import pytest

from app.answer_grounding import evaluate_answer_grounding


class FakeTextBlock:
    """Simulate a text block returned by the Anthropic API."""

    type = "text"

    def __init__(self, text):
        self.text = text


class FakeMessages:
    """Simulate the Anthropic messages API."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return type(
            "FakeResponse",
            (),
            {
                "content": [
                    FakeTextBlock(self.response_text)
                ]
            },
        )()


class FakeClient:
    """Simulate an Anthropic client without making API requests."""

    def __init__(self, response_text):
        self.messages = FakeMessages(response_text)


def sample_chunks():
    return [
        {
            "source": "oomkilled.md",
            "section": "Overview",
            "content": (
                "Exit code 137 indicates that a process was terminated "
                "by SIGKILL. Memory exhaustion is one possible cause."
            ),
            "score": 0.85,
        }
    ]


def test_evaluate_answer_grounding_accepts_supported_claims():
    response = {
        "claims": [
            {
                "claim": "Exit code 137 indicates SIGKILL.",
                "verdict": "supported",
                "reason": "The evidence explicitly describes exit code 137.",
            },
            {
                "claim": "Memory exhaustion is one possible cause.",
                "verdict": "supported",
                "reason": "The evidence identifies memory exhaustion as a cause.",
            },
        ]
    }

    client = FakeClient(json.dumps(response))

    result = evaluate_answer_grounding(
        question="What does exit code 137 mean?",
        answer=(
            "Exit code 137 indicates SIGKILL. "
            "Memory exhaustion is one possible cause."
        ),
        retrieved_chunks=sample_chunks(),
        client=client,
    )

    assert result["grounded"] is True
    assert result["claim_count"] == 2
    assert result["supported_claim_count"] == 2
    assert result["unsupported_claims"] == []
    assert result["contradicted_claims"] == []
    assert len(client.messages.calls) == 1


def test_evaluate_answer_grounding_flags_unsupported_claims():
    response = {
        "claims": [
            {
                "claim": "Exit code 137 indicates SIGKILL.",
                "verdict": "supported",
                "reason": "The evidence supports this interpretation.",
            },
            {
                "claim": "The database password is incorrect.",
                "verdict": "unsupported",
                "reason": "The evidence says nothing about database passwords.",
            },
        ]
    }

    client = FakeClient(json.dumps(response))

    result = evaluate_answer_grounding(
        question="What does exit code 137 mean?",
        answer=(
            "Exit code 137 indicates SIGKILL. "
            "The database password is incorrect."
        ),
        retrieved_chunks=sample_chunks(),
        client=client,
    )

    assert result["grounded"] is False
    assert result["claim_count"] == 2
    assert result["supported_claim_count"] == 1
    assert len(result["unsupported_claims"]) == 1
    assert (
        result["unsupported_claims"][0]["claim"]
        == "The database password is incorrect."
    )


def test_evaluate_answer_grounding_flags_contradicted_claims():
    response = {
        "claims": [
            {
                "claim": "Exit code 137 indicates a normal successful exit.",
                "verdict": "contradicted",
                "reason": "The evidence associates exit code 137 with SIGKILL.",
            }
        ]
    }

    client = FakeClient(json.dumps(response))

    result = evaluate_answer_grounding(
        question="What does exit code 137 mean?",
        answer="Exit code 137 indicates a normal successful exit.",
        retrieved_chunks=sample_chunks(),
        client=client,
    )

    assert result["grounded"] is False
    assert len(result["contradicted_claims"]) == 1
    assert result["unsupported_claims"] == []


def test_evaluate_answer_grounding_rejects_invalid_json():
    client = FakeClient("This is not valid JSON.")

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        evaluate_answer_grounding(
            question="What does exit code 137 mean?",
            answer="Exit code 137 indicates SIGKILL.",
            retrieved_chunks=sample_chunks(),
            client=client,
        )


def test_evaluate_answer_grounding_rejects_invalid_verdict():
    response = {
        "claims": [
            {
                "claim": "Exit code 137 indicates SIGKILL.",
                "verdict": "probably_supported",
                "reason": "The evaluator is uncertain.",
            }
        ]
    }

    client = FakeClient(json.dumps(response))

    with pytest.raises(
        ValueError,
        match="invalid verdict",
    ):
        evaluate_answer_grounding(
            question="What does exit code 137 mean?",
            answer="Exit code 137 indicates SIGKILL.",
            retrieved_chunks=sample_chunks(),
            client=client,
        )


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("", "Exit code 137 indicates SIGKILL."),
        ("   ", "Exit code 137 indicates SIGKILL."),
        ("What does exit code 137 mean?", ""),
        ("What does exit code 137 mean?", "   "),
    ],
)
def test_evaluate_answer_grounding_rejects_empty_input(
    question,
    answer,
):
    client = FakeClient('{"claims": []}')

    with pytest.raises(ValueError):
        evaluate_answer_grounding(
            question=question,
            answer=answer,
            retrieved_chunks=sample_chunks(),
            client=client,
        )

    assert client.messages.calls == []


def test_evaluate_answer_grounding_rejects_empty_evidence():
    client = FakeClient('{"claims": []}')

    with pytest.raises(
        ValueError,
        match="Retrieved evidence cannot be empty",
    ):
        evaluate_answer_grounding(
            question="What does exit code 137 mean?",
            answer="Exit code 137 indicates SIGKILL.",
            retrieved_chunks=[],
            client=client,
        )

    assert client.messages.calls == []


def test_evaluate_answer_grounding_rejects_evidence_without_content():
    client = FakeClient('{"claims": []}')

    chunks = [
        {
            "source": "oomkilled.md",
            "section": "Overview",
            "content": "   ",
        }
    ]

    with pytest.raises(
        ValueError,
        match="no usable content",
    ):
        evaluate_answer_grounding(
            question="What does exit code 137 mean?",
            answer="Exit code 137 indicates SIGKILL.",
            retrieved_chunks=chunks,
            client=client,
        )

    assert client.messages.calls == []


def test_evaluate_answer_grounding_marks_empty_claims_inconclusive():
    client = FakeClient(json.dumps({"claims": []}))

    result = evaluate_answer_grounding(
        question="What does exit code 137 mean?",
        answer="Exit code 137 indicates SIGKILL.",
        retrieved_chunks=sample_chunks(),
        client=client,
    )

    assert result["grounded"] is None
    assert result["evaluation_status"] == "inconclusive"
    assert result["claim_count"] == 0
    assert result["supported_claim_count"] == 0
    assert result["unsupported_claims"] == []
    assert result["contradicted_claims"] == []
