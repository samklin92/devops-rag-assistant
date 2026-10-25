from app.agent_state import AgentState
from app.agent_loop import run_agent


def test_agent_runs_multiple_diagnostic_steps():
    state = AgentState(task="Investigate ImagePullBackOff")

    tools = {
        "inspect_pod_events": lambda: {
            "reason": "FailedToRetrieveImagePullSecret"
        },
        "inspect_image_pull_secret_configuration": lambda: {
            "reason": "AuthenticationConfigurationRequiresReview",
            "terminal": True,
        },
    }

    response = run_agent(state, tools)

    assert response["status"] == "completed"
    assert response["steps"] == 2
    assert state.tool_calls == 2
    assert len(state.findings) == 2
    assert state.next_action is None


def test_agent_rejects_unregistered_tools():
    state = AgentState(task="Investigate ImagePullBackOff")

    response = run_agent(state, {})

    assert response["status"] == "blocked"
    assert "Tool is not registered" in response["reason"]
    assert state.tool_calls == 0


def test_agent_stops_after_tool_failure():
    state = AgentState(task="Investigate ImagePullBackOff")

    def failing_tool():
        raise RuntimeError("Diagnostic unavailable")

    response = run_agent(
        state,
        {"inspect_pod_events": failing_tool},
    )

    assert response["status"] == "failed"
    assert response["reason"] == "Diagnostic unavailable"
    assert state.tool_calls == 1


def test_agent_stops_when_action_repeats():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "previous_check",
        "result": {"reason": "UnknownError"},
    })

    response = run_agent(
        state,
        {"gather_additional_diagnostics": lambda: {"reason": "UnknownError"}},
    )

    assert response["status"] == "needs_review"
    assert "Repeated action detected" in response["reason"]
    assert response["steps"] == 1

def test_repeated_action_returns_recommendation():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "previous_check",
        "result": {"reason": "UnknownError"},
    })

    response = run_agent(
        state,
        {"gather_additional_diagnostics": lambda: {"reason": "UnknownError"}},
    )

    assert response["status"] == "needs_review"
    assert response["next_step"]
    assert "findings" in response["next_step"].lower()


def test_unregistered_tool_returns_recommendation():
    state = AgentState(task="Investigate ImagePullBackOff")

    response = run_agent(state, {})

    assert response["status"] == "blocked"
    assert response["next_step"]
    assert "tool" in response["next_step"].lower()


def test_failed_tool_returns_recommendation():
    state = AgentState(task="Investigate ImagePullBackOff")

    def failing_tool():
        raise RuntimeError("Diagnostic unavailable")

    response = run_agent(
        state,
        {"inspect_pod_events": failing_tool},
    )

    assert response["status"] == "failed"
    assert response["next_step"]
    assert "failed diagnostic tool" in response["next_step"].lower()

def test_stalled_agent_recommends_using_available_knowledge():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "UnknownError"},
    })
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Verify the image name and tag, then check registry credentials.",
        "score": 0.85,
    })

    response = run_agent(
        state,
        {"gather_additional_diagnostics": lambda: {"reason": "UnknownError"}},
    )

    assert response["status"] == "needs_review"
    assert response["next_step"]
    assert "imagepullbackoff.md" in response["next_step"]

def test_agent_recommends_documentation_when_no_knowledge_is_available():
    state = AgentState(task="Investigate an unknown Kubernetes issue")

    response = run_agent(
        state,
        {},
        embedding_model=None,
    )

    assert response["status"] == "blocked"
    assert response["next_step"]
    assert "no relevant rag knowledge" in response["next_step"].lower()


def test_agent_stops_when_knowledge_retrieval_fails(monkeypatch):
    from app import agent_loop

    state = AgentState(task="Investigate ImagePullBackOff")

    def failing_retrieval(**kwargs):
        raise RuntimeError("Vector index unavailable")

    monkeypatch.setattr(
        agent_loop,
        "retrieve_agent_knowledge",
        failing_retrieval,
    )

    response = run_agent(
        state,
        {"inspect_pod_events": lambda: {"reason": "UnknownError"}},
        embedding_model="fake-model",
    )

    assert response["status"] == "failed"
    assert "Vector index unavailable" in response["reason"]
    assert response["steps"] == 0
    assert "retrieval error" in response["next_step"].lower()
    assert state.tool_calls == 0



def test_agent_generates_final_response_when_investigation_completes():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Check the image name and registry credentials.",
        "score": 0.85,
    })

    tools = {
        "inspect_pod_events": lambda: {
            "reason": "FailedToRetrieveImagePullSecret",
            "terminal": True,
        },
    }

    class FakeClient:
        pass

    def fake_answer_generator(**kwargs):
        return "Check the image pull secret configuration."

    from app.agent_loop import run_agent
    from app.agent_response import generate_agent_response

    # Replace response generation with a predictable test implementation.
    import app.agent_loop as agent_loop

    original_generator = agent_loop.generate_agent_response

    def fake_response_generator(state, client):
        return generate_agent_response(
            state,
            client,
            answer_generator=fake_answer_generator,
        )

    agent_loop.generate_agent_response = fake_response_generator

    try:
        response = run_agent(state, tools, client=FakeClient())
    finally:
        agent_loop.generate_agent_response = original_generator

    assert response["status"] == "completed"
    assert response["final_response"]["answer"] == (
        "Check the image pull secret configuration."
    )
    assert response["final_response"]["sources"][0]["source"] == (
        "imagepullbackoff.md"
    )


def test_agent_stops_when_diagnostics_remain_inconclusive():
    state = AgentState(task="Investigate an unknown Kubernetes issue")
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Check image names and registry credentials.",
        "score": 0.85,
    })

    tools = {
        "inspect_pod_events": lambda: {"reason": "UnknownError"},
        "gather_additional_diagnostics": lambda: {
            "reason": "UnknownError",
        },
    }

    response = run_agent(state, tools)

    assert response["status"] == "needs_review"
    assert "Repeated action detected" in response["reason"]
    assert response["next_step"]
    assert "imagepullbackoff.md" in response["next_step"]
    assert state.tool_calls == 2


def test_controller_never_exceeds_tool_call_limit(monkeypatch):
    from app import agent_loop
    from app.tool_executor import MAX_TOOL_CALLS

    state = AgentState(task="Investigate an unknown issue")
    actions = iter(
        [f"diagnostic_{number}" for number in range(MAX_TOOL_CALLS + 2)]
    )

    monkeypatch.setattr(
        agent_loop,
        "decide_next_action",
        lambda current_state: next(actions),
    )

    tools = {
        f"diagnostic_{number}": lambda: {"reason": "UnknownError"}
        for number in range(MAX_TOOL_CALLS + 2)
    }

    response = run_agent(
        state,
        tools,
        max_steps=MAX_TOOL_CALLS + 2,
    )

    assert response["status"] == "needs_review"
    assert response["steps"] == MAX_TOOL_CALLS
    assert state.tool_calls == MAX_TOOL_CALLS


def test_invalid_max_steps_is_rejected():
    import pytest

    state = AgentState(task="Investigate ImagePullBackOff")

    with pytest.raises(ValueError, match="positive integer"):
        run_agent(state, {}, max_steps=0)


def test_response_generation_failure_returns_recommendation(monkeypatch):
    from app import agent_loop

    state = AgentState(task="Investigate ImagePullBackOff")
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Check registry credentials.",
        "score": 0.85,
    })

    def failing_response(**kwargs):
        raise RuntimeError("LLM service unavailable")

    monkeypatch.setattr(
        agent_loop,
        "generate_agent_response",
        failing_response,
    )

    response = run_agent(
        state,
        {
            "inspect_pod_events": lambda: {
                "reason": "AuthenticationError",
                "terminal": True,
            }
        },
        client=object(),
    )

    assert response["status"] == "needs_review"
    assert "LLM service unavailable" in response["reason"]
    assert response["final_response"] is None
    assert "retry response generation" in response["next_step"].lower()
    assert state.tool_calls == 1


def test_controller_blocks_mutating_tool_without_approval():
    from app.agent_loop import run_agent
    from app.agent_state import AgentState
    from app.tool_executor import ToolDefinition

    state = AgentState(task="Investigate an unknown issue")
    executed = []

    tools = {
        "inspect_pod_events": ToolDefinition(
            handler=lambda: executed.append(True) or {"reason": "UnknownError"},
            read_only=False,
        )
    }

    response = run_agent(state, tools)

    assert response["status"] == "blocked"
    assert "approval required" in response["reason"].lower()
    assert executed == []
    assert state.tool_calls == 0


def test_controller_allows_approved_mutating_tool():
    from app.agent_loop import run_agent
    from app.agent_state import AgentState
    from app.tool_executor import ToolDefinition

    state = AgentState(task="Investigate an unknown issue")
    tools = {
        "inspect_pod_events": ToolDefinition(
            handler=lambda: {"reason": "InvestigationComplete", "terminal": True},
            read_only=False,
        )
    }

    response = run_agent(
        state,
        tools,
        approved_actions={"inspect_pod_events"},
    )

    assert response["status"] == "completed"
    assert state.tool_calls == 1
