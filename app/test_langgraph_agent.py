from unittest.mock import patch

from app.langgraph_agent import retrieve_knowledge_node

from app.langgraph_agent import select_diagnostic_action_node

from app.langgraph_agent import execute_diagnostic_tool_node

from app.langgraph_agent import evaluate_evidence_node

from unittest.mock import patch

from app.langgraph_agent import (
    retrieve_knowledge_node,
    select_diagnostic_action_node,
    execute_diagnostic_tool_node,
    evaluate_evidence_node,
    build_devops_graph,
)

def test_retrieve_knowledge_node():
    state = {
        "task": "My Kubernetes pod shows ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 0,
        "next_action": None,
    }

    expected_knowledge = [
        {
            "content": "Check image name, tag and registry credentials.",
            "source": "image-pull-backoff.md",
        }
    ]

    with patch(
        "app.langgraph_agent.retrieve_agent_knowledge",
        return_value=expected_knowledge,
    ) as mock_retrieve:

        result = retrieve_knowledge_node(
            state,
            embedding_model="fake-model",
        )

    assert result["knowledge"] == expected_knowledge

    mock_retrieve.assert_called_once_with(
        question="My Kubernetes pod shows ImagePullBackOff.",
        embedding_model="fake-model",
        top_k=3,
    )

def test_select_diagnostic_action_node():
    state = {
        "task": "My pod shows ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [
            {
                "result": {
                    "reason": "FailedToRetrieveImagePullSecret"
                }
            }
        ],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 0,
        "next_action": None,
    }

    result = select_diagnostic_action_node(state)

    assert result["next_action"] == (
        "inspect_image_pull_secret_configuration"
    )

def test_execute_diagnostic_tool_node():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 0,
        "next_action": "inspect_pod_events",
    }

    tools = {
        "inspect_pod_events": lambda: {
            "reason": "ImageNotFound",
            "message": "Container image does not exist.",
        }
    }

    result = execute_diagnostic_tool_node(state, tools)

    assert result["status"] == "running"
    assert result["tool_calls"] == 1
    assert len(result["findings"]) == 1
    assert result["findings"][0]["result"]["reason"] == "ImageNotFound"

"""Tests for the evidence-evaluation node of the investigation graph."""

# Adjust this import to the module where evaluate_evidence_node is defined.
from app.langgraph_agent import evaluate_evidence_node


def test_evaluate_evidence_completes_terminal_investigation():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [
            {"tool": "inspect_pod_events", "result": {"terminal": True}}
        ],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 1,
        "next_action": None,
    }

    result = evaluate_evidence_node(state)

    assert result == {
        "status": "completed",
        "next_action": None,
    }


def test_evaluate_evidence_continues_investigation():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [
            {"tool": "inspect_pod_events", "result": {"reason": "ImageNotFound"}}
        ],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 1,
        "next_action": "inspect_image_name_and_tag",
    }

    result = evaluate_evidence_node(state)

    assert result == {"status": "running"}


def test_evaluate_evidence_stops_after_maximum_tool_calls():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [
            {"tool": "inspect_pod_events", "result": {"reason": "ImageNotFound"}}
        ],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 5,
        "next_action": "gather_additional_diagnostics",
    }

    result = evaluate_evidence_node(state)

    assert result == {
        "status": "needs_review",
        "next_action": None,
    }


def test_evaluate_evidence_stops_when_findings_are_empty():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "running",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 1,
        "next_action": None,
    }

    result = evaluate_evidence_node(state)

    assert result == {
        "status": "needs_review",
        "next_action": None,
    }


def test_evaluate_evidence_preserves_failed_status():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "failed",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 1,
        "next_action": None,
    }

    result = evaluate_evidence_node(state)

    assert result == {
        "status": "failed",
        "next_action": None,
    }


def test_evaluate_evidence_preserves_blocked_status():
    state = {
        "task": "Investigate ImagePullBackOff.",
        "task_id": "test-123",
        "status": "blocked",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 1,
        "next_action": None,
    }

    result = evaluate_evidence_node(state)

    assert result == {
        "status": "blocked",
        "next_action": None,
    }

def test_complete_devops_graph_workflow():

    tools = {
        "inspect_pod_events": lambda: {
            "reason": "ImageNotFound",
            "message": "The requested container image was not found.",
            "terminal": False,
        },
        "inspect_image_name_and_tag": lambda: {
            "reason": "ImageNameAndTagConfirmed",
            "message": "The image name contains an incorrect tag.",
            "terminal": True,
        },
    }

    knowledge = [
        {
            "content": "Check the image name and tag.",
            "source": "image-pull-backoff.md",
            "section": "Possible Causes",
        }
    ]

    expected_response = {
        "answer": "The ImagePullBackOff is caused by an incorrect image tag.",
        "sources": [
            {
                "source": "image-pull-backoff.md",
                "section": "Possible Causes",
            }
        ],
        "grounded_in_retrieved_knowledge": True,
    }

    with (
        patch(
            "app.langgraph_agent.retrieve_agent_knowledge",
            return_value=knowledge,
        ),
        patch(
            "app.langgraph_agent.generate_agent_response",
            return_value=expected_response,
        ),
    ):

        graph = build_devops_graph(
            embedding_model="fake-model",
            tools=tools,
            client="fake-client",
        )

        initial_state = {
            "task": "Investigate ImagePullBackOff.",
            "task_id": "test-graph-123",
            "status": "pending",
            "findings": [],
            "actions_taken": [],
            "knowledge": [],
            "tool_calls": 0,
            "next_action": None,
        }

        result = graph.invoke(initial_state)

    assert result["status"] == "completed"
    assert result["tool_calls"] == 2
    assert len(result["findings"]) == 2
    assert result["response"] == expected_response["answer"]
    assert result["grounded_in_retrieved_knowledge"] is True

def test_graph_stops_at_maximum_tool_calls():
    tools = {
        "inspect_pod_events": lambda: {
            "reason": "Still investigating",
            "terminal": False,
        }
    }

    with (
        patch("app.langgraph_agent.retrieve_agent_knowledge", return_value=[]),
        patch(
            "app.langgraph_agent.generate_agent_response",
            return_value={
                "answer": "Investigation requires review.",
                "sources": [],
                "grounded_in_retrieved_knowledge": False,
            },
        ),
    ):
        graph = build_devops_graph(
            embedding_model="fake-model",
            tools=tools,
            client="fake-client",
        )

        result = graph.invoke({
            "task": "Investigate ImagePullBackOff.",
            "task_id": "limit-test",
            "status": "pending",
            "findings": [],
            "actions_taken": [],
            "knowledge": [],
            "tool_calls": 4,
            "next_action": None,
        })

    assert result["tool_calls"] == 5
    assert result["status"] == "needs_review"

def test_graph_blocks_unapproved_modifying_tool():
    from app.tool_executor import ToolDefinition

    tools = {
        "inspect_pod_events": ToolDefinition(
            handler=lambda: {"terminal": True},
            read_only=False,
        )
    }

    with (
        patch("app.langgraph_agent.retrieve_agent_knowledge", return_value=[]),
        patch(
            "app.langgraph_agent.generate_agent_response",
            return_value={
                "answer": "Operation blocked pending approval.",
                "sources": [],
                "grounded_in_retrieved_knowledge": False,
            },
        ),
    ):
        graph = build_devops_graph(
            embedding_model="fake-model",
            tools=tools,
            client="fake-client",
        )

        result = graph.invoke({
            "task": "Investigate ImagePullBackOff.",
            "task_id": "approval-test",
            "status": "pending",
            "findings": [],
            "actions_taken": [],
            "knowledge": [],
            "tool_calls": 0,
            "next_action": None,
        })

    assert result["status"] == "blocked"
    assert result["tool_calls"] == 0

    