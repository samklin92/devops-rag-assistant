"""Tests for the Kubernetes MCP server."""

import asyncio

import pytest

from app.mcp_server.server import mcp

from types import SimpleNamespace

from app.mcp_server.adapter import build_mcp_tools

import app.langgraph_agent as langgraph_agent

from app.langgraph_agent import build_devops_graph_async

def test_mcp_server_registers_all_diagnostic_tools():

    result = asyncio.run(mcp.list_tools())

    registered_tool_names = {
        tool.name for tool in result
    }

    expected_tools = {
        "inspect_pod",
        "get_pod_events",
        "get_container_logs",
        "check_deployment_status",
    }

    assert registered_tool_names == expected_tools

def test_adapter_inspects_pod_events():

    class FakeSession:

        async def call_tool(self, name, arguments):

            assert name == "get_pod_events"
            assert arguments == {"pod_name": "inventory-api"}

            return SimpleNamespace(
                is_error=False,
                structured_content={
                    "events": [
                        {
                            "type": "Warning",
                            "reason": "BackOff",
                            "message": "Back-off restarting failed container",
                        }
                    ]
                },
            )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
    )

    result = asyncio.run(
        tools["inspect_pod_events"]()
    )

    assert result["events"][0]["reason"] == "BackOff"

def test_adapter_gathers_additional_diagnostics():

    class FakeSession:

        async def call_tool(self, name, arguments):

            responses = {
                "inspect_pod": {
                    "pod_name": "inventory-api",
                    "status": "CrashLoopBackOff",
                },
                "get_container_logs": {
                    "logs": ["ERROR: Database connection refused."]
                },
                "check_deployment_status": {
                    "desired_replicas": 3,
                    "ready_replicas": 1,
                    "status": "Degraded",
                },
            }

            assert name in responses

            return SimpleNamespace(
                is_error=False,
                structured_content=responses[name],
            )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
    )

    result = asyncio.run(
        tools["gather_additional_diagnostics"]()
    )

    assert result["pod"]["status"] == "CrashLoopBackOff"
    assert result["logs"]["logs"][0] == (
        "ERROR: Database connection refused."
    )
    assert result["deployment"]["status"] == "Degraded"
    assert result["terminal"] is True

def test_adapter_raises_error_when_mcp_tool_fails():

    class FakeSession:

        async def call_tool(self, name, arguments):

            return SimpleNamespace(
                is_error=True,
                structured_content=None,
            )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
    )

    with pytest.raises(
        RuntimeError,
        match="MCP failed to retrieve pod events.",
    ):
        asyncio.run(
            tools["inspect_pod_events"]()
        )
def test_mcp_tools_integrate_with_async_langgraph(monkeypatch):

    calls = []

    class FakeSession:

        async def call_tool(self, name, arguments):

            calls.append(name)

            responses = {
                "get_pod_events": {
                    "events": [
                        {
                            "type": "Warning",
                            "reason": "BackOff",
                            "message": "Back-off restarting failed container",
                        }
                    ]
                },
                "inspect_pod": {
                    "pod_name": "inventory-api",
                    "status": "CrashLoopBackOff",
                },
                "get_container_logs": {
                    "logs": [
                        "ERROR: Database connection refused."
                    ]
                },
                "check_deployment_status": {
                    "desired_replicas": 3,
                    "ready_replicas": 1,
                    "status": "Degraded",
                },
            }

            return SimpleNamespace(
                is_error=False,
                structured_content=responses[name],
            )

    # Mock RAG retrieval to avoid loading the embedding model.
    monkeypatch.setattr(
        langgraph_agent,
        "retrieve_agent_knowledge",
        lambda **kwargs: [
            {
                "source": "crashloopbackoff.md",
                "section": "Possible Causes",
                "content": "Database connectivity problems can cause crashes.",
            }
        ],
    )

    # Mock response generation to avoid calling Claude's API.
    monkeypatch.setattr(
        langgraph_agent,
        "generate_agent_response",
        lambda state, client: {
            "answer": "The pod is experiencing database connectivity problems.",
            "sources": state.knowledge,
            "grounded_in_retrieved_knowledge": True,
        },
    )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
    )

    graph = build_devops_graph_async(
        embedding_model=object(),
        tools=tools,
        client=object(),
    )

    initial_state = {
        "task": "Investigate why inventory-api is in CrashLoopBackOff",
        "task_id": "mcp-integration-test",
        "status": "pending",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 0,
        "next_action": None,
        "response": "",
        "sources": [],
        "grounded_in_retrieved_knowledge": False,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert result["status"] == "completed"
    assert result["tool_calls"] == 2

    assert calls == [
        "get_pod_events",
        "inspect_pod",
        "get_container_logs",
        "check_deployment_status",
    ]

    assert result["response"] == (
        "The pod is experiencing database connectivity problems."
    )

    assert result["grounded_in_retrieved_knowledge"] is True
