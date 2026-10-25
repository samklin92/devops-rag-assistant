from app.agent_state import AgentState


def test_agent_state_initializes_correctly():
    state = AgentState(task="Investigate ImagePullBackOff")

    assert state.task == "Investigate ImagePullBackOff"
    assert state.task_id
    assert state.status == "pending"
    assert state.findings == []
    assert state.actions_taken == []
    assert state.tool_calls == 0
    assert state.next_action is None


def test_each_task_has_a_unique_id():
    first = AgentState(task="Investigate pod A")
    second = AgentState(task="Investigate pod B")

    assert first.task_id != second.task_id


def test_tasks_have_independent_lists():
    first = AgentState(task="Investigate pod A")
    second = AgentState(task="Investigate pod B")

    first.findings.append({"observation": "Authentication failed"})
    first.actions_taken.append({"action": "Inspect pod events"})

    assert second.findings == []
    assert second.actions_taken == []
