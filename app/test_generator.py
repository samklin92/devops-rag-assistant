import pytest

from app.generator import generate_answer


class FakeTextBlock:
    """Simulate a text block returned by the Anthropic API."""

    type = "text"

    def __init__(self, text):
        self.text = text


class FakeMessages:
    """Simulate the Anthropic messages API."""

    def __init__(self, response_text="Check the pod logs and events."):
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

    def __init__(self, response_text="Check the pod logs and events."):
        self.messages = FakeMessages(response_text)


def sample_chunks():
    return [
        {
            "source": "crashloopbackoff.md",
            "section": "Diagnostic Commands",
            "content": (
                "Use kubectl logs to inspect the previous container "
                "instance and kubectl describe pod to inspect events."
            ),
            "score": 0.85,
        }
    ]


def test_generate_answer_returns_llm_response():
    client = FakeClient(
        "Inspect the previous container logs and pod events."
    )

    answer = generate_answer(
        question="Why does my pod keep restarting?",
        retrieved_chunks=sample_chunks(),
        client=client,
    )

    assert answer == (
        "Inspect the previous container logs and pod events."
    )
    assert len(client.messages.calls) == 1


def test_generate_answer_sends_question_and_context():
    client = FakeClient()

    generate_answer(
        question="Why does my pod keep restarting?",
        retrieved_chunks=sample_chunks(),
        client=client,
    )

    request = client.messages.calls[0]
    user_message = request["messages"][0]["content"]

    assert "Why does my pod keep restarting?" in user_message
    assert "crashloopbackoff.md" in user_message
    assert "kubectl logs" in user_message


def test_generate_answer_rejects_empty_question():
    client = FakeClient()

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        generate_answer(
            question="   ",
            retrieved_chunks=sample_chunks(),
            client=client,
        )

    assert client.messages.calls == []


def test_generate_answer_handles_no_retrieved_chunks():
    client = FakeClient()

    answer = generate_answer(
        question="How do I fix this issue?",
        retrieved_chunks=[],
        client=client,
    )

    assert "could not find relevant information" in answer.lower()
    assert client.messages.calls == []


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_generate_answer_rejects_invalid_max_tokens(max_tokens):
    client = FakeClient()

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        generate_answer(
            question="Why is my pod restarting?",
            retrieved_chunks=sample_chunks(),
            client=client,
            max_tokens=max_tokens,
        )

    assert client.messages.calls == []


def test_generate_answer_handles_empty_retrieved_content():
    client = FakeClient()

    chunks = [
        {
            "source": "crashloopbackoff.md",
            "section": "Symptoms",
            "content": "   ",
        }
    ]

    answer = generate_answer(
        question="Why is my pod restarting?",
        retrieved_chunks=chunks,
        client=client,
    )

    assert "could not find usable evidence" in answer.lower()
    assert client.messages.calls == []

def test_context_budget_includes_complete_chunks_that_fit():
    chunks = [
        {
            "source": "crashloopbackoff.md",
            "section": "Diagnostics",
            "content": "A" * 100,
            "score": 0.95,
        },
        {
            "source": "database-connectivity.md",
            "section": "Troubleshooting",
            "content": "B" * 100,
            "score": 0.80,
        },
    ]

    fake_client = FakeClient()

    generate_answer(
        question="How do I troubleshoot this issue?",
        retrieved_chunks=chunks,
        client=fake_client,
        max_context_chars=180,
    )

    sent_prompt = fake_client.messages.calls[0]["messages"][0]["content"]

    assert "Source: crashloopbackoff.md" in sent_prompt
    assert "Section: Diagnostics" in sent_prompt
    assert "A" * 100 in sent_prompt

    # The second chunk must not be partially included.
    assert "Source: database-connectivity.md" not in sent_prompt
    assert "B" * 100 not in sent_prompt


def test_context_budget_returns_safe_message_when_no_chunk_fits():
    chunks = [
        {
            "source": "crashloopbackoff.md",
            "section": "Diagnostics",
            "content": "A" * 100,
            "score": 0.95,
        }
    ]

    fake_client = FakeClient()

    answer = generate_answer(
        question="How do I troubleshoot this issue?",
        retrieved_chunks=chunks,
        client=fake_client,
        max_context_chars=50,
    )

    assert "could not find usable evidence" in answer.lower()
    assert fake_client.messages.calls == []


def test_context_budget_must_be_positive():
    fake_client = FakeClient()

    with pytest.raises(ValueError, match="max_context_chars"):
        generate_answer(
            question="How do I troubleshoot this issue?",
            retrieved_chunks=sample_chunks(),
            client=fake_client,
            max_context_chars=0,
        )
