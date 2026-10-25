"""Tests for controlled diagnostic tool execution."""

import pytest

from app.agent_state import AgentState
from app.tool_executor import MAX_TOOL_CALLS, execute_tool


def test_successful_tool_call_records_result():
    state = AgentState(task="Investigate ImagePullBackOff")

    response = execute_tool(
        state,
        "inspect_pod_events",
        lambda: {"reason": "FailedToRetrieveImagePullSecret"},
    )

    assert response["success"] is True
    assert state.tool_calls == 1
    assert state.findings[0]["result"]["reason"] == (
        "FailedToRetrieveImagePullSecret"
    )
    assert state.actions_taken[0]["status"] == "completed"


def test_failed_tool_call_is_recorded():
    state = AgentState(task="Investigate ImagePullBackOff")

    def failing_tool():
        raise RuntimeError("Diagnostic tool unavailable")

    response = execute_tool(state, "inspect_pod_events", failing_tool)

    assert response["success"] is False
    assert response["error"] == "Diagnostic tool unavailable"
    assert state.tool_calls == 1
    assert state.actions_taken[-1]["status"] == "failed"
    assert state.findings == []


def test_tool_call_limit_is_enforced():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.tool_calls = MAX_TOOL_CALLS

    def should_not_run():
        raise AssertionError("Tool must not execute after the limit")

    response = execute_tool(state, "inspect_pod_events", should_not_run)

    assert response["success"] is False
    assert response["error"] == "Maximum tool-call limit reached."
    assert state.tool_calls == MAX_TOOL_CALLS


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        [],
        "unexpected result",
        42,
    ],
)
def test_invalid_tool_results_are_rejected(invalid_result):
    state = AgentState(task="Investigate ImagePullBackOff")

    response = execute_tool(
        state,
        "inspect_pod_events",
        lambda: invalid_result,
    )

    assert response["success"] is False
    assert "invalid result" in response["error"].lower()
    assert state.tool_calls == 1
    assert state.actions_taken[-1]["status"] == "failed"
    assert state.findings == []


def test_mutating_tool_is_blocked_without_approval():
    from app.tool_executor import ToolDefinition

    state = AgentState(task="Restart a production workload")
    executed = []

    tool = ToolDefinition(
        handler=lambda: executed.append(True) or {"status": "restarted"},
        read_only=False,
    )

    response = execute_tool(state, "restart_workload", tool)

    assert response["success"] is False
    assert response["blocked"] is True
    assert "approval required" in response["error"].lower()
    assert executed == []
    assert state.tool_calls == 0


def test_mutating_tool_runs_with_explicit_approval():
    from app.tool_executor import ToolDefinition

    state = AgentState(task="Restart a production workload")
    tool = ToolDefinition(
        handler=lambda: {"status": "restarted"},
        read_only=False,
    )

    response = execute_tool(
        state,
        "restart_workload",
        tool,
        approved=True,
    )

    assert response["success"] is True
    assert state.tool_calls == 1
    assert state.findings[-1]["result"]["status"] == "restarted"
