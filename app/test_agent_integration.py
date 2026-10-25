"""Integration tests for the DevOps agent investigation workflow."""

from app.agent_loop import run_agent
from app.agent_state import AgentState
from app.tool_executor import ToolDefinition


def test_complete_investigation_from_rag_to_final_response(monkeypatch):
    from app import agent_loop

    state = AgentState(
        task="Investigate a Kubernetes ImagePullBackOff error"
    )

    # Simulate retrieving relevant documentation from the RAG system.
    retrieved_knowledge = [
        {
            "source": "imagepullbackoff.md",
            "section": "Troubleshooting Steps",
            "content": (
                "Check image names, registry credentials, "
                "and image pull secret configuration."
            ),
            "score": 0.91,
        }
    ]

    monkeypatch.setattr(
        agent_loop,
        "retrieve_agent_knowledge",
        lambda **kwargs: retrieved_knowledge,
    )

    # Simulate generating a response using the findings and RAG evidence.
    monkeypatch.setattr(
        agent_loop,
        "generate_agent_response",
        lambda **kwargs: {
            "answer": (
                "The diagnostic findings indicate an image pull "
                "authentication problem. Review the image pull secret."
            ),
            "sources": [
                {
                    "source": "imagepullbackoff.md",
                    "section": "Troubleshooting Steps",
                    "score": 0.91,
                }
            ],
            "grounded_in_retrieved_knowledge": True,
        },
    )

    # Simulate read-only diagnostic tools.
    tools = {
        "inspect_pod_events": ToolDefinition(
            handler=lambda: {
                "reason": "FailedToRetrieveImagePullSecret"
            },
            read_only=True,
        ),
        "inspect_image_pull_secret_configuration": ToolDefinition(
            handler=lambda: {
                "reason": "AuthenticationConfigurationRequiresReview",
                "terminal": True,
            },
            read_only=True,
        ),
    }

    response = run_agent(
        state,
        tools,
        embedding_model=object(),
        client=object(),
    )

    assert response["status"] == "completed"
    assert response["steps"] == 2
    assert state.tool_calls == 2
    assert len(state.findings) == 2
    assert state.knowledge == retrieved_knowledge

    final_response = response["final_response"]
    assert "authentication problem" in final_response["answer"].lower()
    assert final_response["sources"][0]["source"] == "imagepullbackoff.md"
    assert final_response["grounded_in_retrieved_knowledge"] is True


def test_integration_blocks_unapproved_infrastructure_change():
    state = AgentState(task="Investigate and remediate a production incident")

    tools = {
        "inspect_pod_events": ToolDefinition(
            handler=lambda: {
                "reason": "UnknownError",
                "terminal": True,
            },
            read_only=False,
        )
    }

    response = run_agent(state, tools)

    assert response["status"] == "blocked"
    assert state.tool_calls == 0
    assert state.findings == []
    assert "approval required" in response["reason"].lower()
