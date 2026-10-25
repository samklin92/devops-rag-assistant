"""Tests for evidence-aware agent response generation."""

from app.agent_response import generate_agent_response
from app.agent_state import AgentState


class FakeClient:
    """Placeholder client; the mocked generator makes no API calls."""


def fake_answer_generator(**kwargs):
    return "Check the image name and verify registry credentials."


def test_response_combines_findings_with_retrieved_knowledge():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "FailedToRetrieveImagePullSecret"},
    })
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Check registry credentials.",
        "score": 0.85,
    })

    captured = {}

    def fake_generator(**kwargs):
        captured.update(kwargs)
        return "Check the image name and verify registry credentials."

    response = generate_agent_response(
        state,
        FakeClient(),
        answer_generator=fake_generator,
    )

    assert "ImagePullBackOff" in captured["question"]
    assert "FailedToRetrieveImagePullSecret" in captured["question"]
    assert captured["retrieved_chunks"] == state.knowledge
    assert response["answer"] == (
        "Check the image name and verify registry credentials."
    )
    assert response["grounded_in_retrieved_knowledge"] is True
    assert response["sources"][0]["source"] == "imagepullbackoff.md"


def test_response_handles_missing_knowledge_without_calling_generator():
    state = AgentState(task="Investigate an unknown issue")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "UnknownError"},
    })

    def unexpected_generator(**kwargs):
        raise AssertionError("Generator should not be called without knowledge")

    response = generate_agent_response(
        state,
        FakeClient(),
        answer_generator=unexpected_generator,
    )

    assert response["sources"] == []
    assert response["grounded_in_retrieved_knowledge"] is False
    assert "no relevant knowledge-base evidence" in response["answer"].lower()


def test_response_deduplicates_source_sections():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.knowledge.extend([
        {
            "source": "imagepullbackoff.md",
            "section": "Troubleshooting",
            "content": "Check credentials.",
            "score": 0.85,
        },
        {
            "source": "imagepullbackoff.md",
            "section": "Troubleshooting",
            "content": "Check image names.",
            "score": 0.80,
        },
    ])

    response = generate_agent_response(
        state,
        FakeClient(),
        answer_generator=fake_answer_generator,
    )

    assert len(response["sources"]) == 1