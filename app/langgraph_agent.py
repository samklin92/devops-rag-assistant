"""LangGraph implementation of the DevOps agent."""

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent_decision import decide_next_action
from app.agent_knowledge import retrieve_agent_knowledge
from app.agent_response import generate_agent_response
from app.agent_state import AgentState
from app.tool_executor import (
    MAX_TOOL_CALLS,
    ToolDefinition,
    execute_tool,
)


class DevOpsGraphState(TypedDict):
    """State shared between LangGraph nodes."""

    task: str
    task_id: str
    status: str
    findings: list[dict[str, Any]]
    actions_taken: list[dict[str, Any]]
    knowledge: list[dict[str, Any]]
    tool_calls: int
    next_action: str | None
    response: str
    sources: list[dict[str, Any]]
    grounded_in_retrieved_knowledge: bool


def to_graph_state(state: AgentState) -> DevOpsGraphState:
    """Convert the existing agent state into LangGraph state."""
    return {
        "task": state.task,
        "task_id": state.task_id,
        "status": state.status,
        "findings": state.findings,
        "actions_taken": state.actions_taken,
        "knowledge": state.knowledge,
        "tool_calls": state.tool_calls,
        "next_action": state.next_action,
    }


def retrieve_knowledge_node(
    state: DevOpsGraphState,
    embedding_model: Any,
) -> dict[str, Any]:
    """Retrieve relevant DevOps knowledge for the investigation."""
    knowledge = retrieve_agent_knowledge(
        question=state["task"],
        embedding_model=embedding_model,
        top_k=3,
    )

    return {
        "knowledge": knowledge,
    }


def select_diagnostic_action_node(
    state: DevOpsGraphState,
) -> dict[str, Any]:
    """Select the next diagnostic action using existing decision logic."""
    agent_state = AgentState(
        task=state["task"],
        task_id=state["task_id"],
        status=state["status"],
        findings=state["findings"],
        actions_taken=state["actions_taken"],
        knowledge=state["knowledge"],
        tool_calls=state["tool_calls"],
        next_action=state["next_action"],
    )

    next_action = decide_next_action(agent_state)

    return {
        "next_action": next_action,
    }


def execute_diagnostic_tool_node(
    state: DevOpsGraphState,
    tools: dict[str, Callable[[], dict[str, Any]] | ToolDefinition],
    approved_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Execute the selected diagnostic tool with existing safety controls."""
    action = state["next_action"]

    if not action:
        return {
            "status": "failed",
            "next_action": None,
        }

    if state["tool_calls"] >= MAX_TOOL_CALLS:
        return {
            "status": "needs_review",
            "next_action": None,
        }

    tool = tools.get(action)

    if tool is None:
        return {
            "status": "blocked",
            "next_action": None,
        }

    approved = approved_actions is not None and action in approved_actions

    agent_state = AgentState(
        task=state["task"],
        task_id=state["task_id"],
        status=state["status"],
        findings=list(state["findings"]),
        actions_taken=list(state["actions_taken"]),
        knowledge=list(state["knowledge"]),
        tool_calls=state["tool_calls"],
        next_action=action,
    )

    result = execute_tool(
        state=agent_state,
        tool_name=action,
        tool=tool,
        approved=approved,
    )

    if not result["success"]:
        # Record execution errors as findings so the final response can
        # explain what failed. Blocked actions are policy decisions, not
        # tool failures, so they are not recorded as findings.
        if not result.get("blocked"):
            agent_state.findings.append(
                {
                    "tool": action,
                    "result": {
                        "error": result.get(
                            "error",
                            "Unknown tool execution error.",
                        ),
                        "execution_failed": True,
                    },
                }
            )

        return {
            "status": "blocked" if result.get("blocked") else "failed",
            "findings": agent_state.findings,
            "actions_taken": agent_state.actions_taken,
            "tool_calls": agent_state.tool_calls,
            "next_action": None,
        }

    return {
        "status": "running",
        "findings": agent_state.findings,
        "actions_taken": agent_state.actions_taken,
        "tool_calls": agent_state.tool_calls,
    }


def evaluate_evidence_node(
    state: DevOpsGraphState,
) -> dict[str, Any]:
    """Evaluate investigation results and determine whether to continue."""
    if state["status"] in {"failed", "blocked", "needs_review"}:
        return {
            "status": state["status"],
            "next_action": None,
        }

    if state["tool_calls"] >= MAX_TOOL_CALLS:
        return {
            "status": "needs_review",
            "next_action": None,
        }

    if not state["findings"]:
        return {
            "status": "needs_review",
            "next_action": None,
        }

    latest_finding = state["findings"][-1]
    result = latest_finding.get("result", {})

    if result.get("terminal") is True:
        return {
            "status": "completed",
            "next_action": None,
        }

    return {
        "status": "running",
    }


def route_after_evaluation(
    state: DevOpsGraphState,
) -> str:
    """Decide whether the investigation should continue or stop."""
    if state["status"] == "running":
        return "continue"

    return "stop"


def generate_final_response_node(
    state: DevOpsGraphState,
    client: Any,
) -> dict[str, Any]:
    """Generate a final response using findings and retrieved knowledge."""
    agent_state = AgentState(
        task=state["task"],
        task_id=state["task_id"],
        status=state["status"],
        findings=state["findings"],
        actions_taken=state["actions_taken"],
        knowledge=state["knowledge"],
        tool_calls=state["tool_calls"],
        next_action=None,
    )

    result = generate_agent_response(
        state=agent_state,
        client=client,
    )

    return {
        "response": result["answer"],
        "sources": result["sources"],
        "grounded_in_retrieved_knowledge": (
            result["grounded_in_retrieved_knowledge"]
        ),
    }


def build_devops_graph(
    embedding_model: Any,
    tools: dict[str, Callable[[], dict[str, Any]] | ToolDefinition],
    client: Any,
    approved_actions: set[str] | None = None,
):
    """Build and compile the LangGraph DevOps investigation workflow."""
    workflow = StateGraph(DevOpsGraphState)

    # Register workflow nodes.
    workflow.add_node(
        "retrieve_knowledge",
        lambda state: retrieve_knowledge_node(
            state,
            embedding_model,
        ),
    )

    workflow.add_node(
        "select_action",
        select_diagnostic_action_node,
    )

    workflow.add_node(
        "execute_tool",
        lambda state: execute_diagnostic_tool_node(
            state,
            tools,
            approved_actions,
        ),
    )

    workflow.add_node(
        "evaluate_evidence",
        evaluate_evidence_node,
    )

    workflow.add_node(
        "generate_response",
        lambda state: generate_final_response_node(state, client),
    )

    # Connect the nodes.
    workflow.add_edge(START, "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "select_action")
    workflow.add_edge("select_action", "execute_tool")
    workflow.add_edge("execute_tool", "evaluate_evidence")

    # Route based on investigation results.
    workflow.add_conditional_edges(
        "evaluate_evidence",
        route_after_evaluation,
        {
            "continue": "select_action",
            "stop": "generate_response",
        },
    )

    workflow.add_edge("generate_response", END)

    return workflow.compile()