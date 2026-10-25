"""Controlled tool execution for the DevOps agent."""

from dataclasses import dataclass
from typing import Any, Callable

from app.agent_state import AgentState


MAX_TOOL_CALLS = 5


@dataclass(frozen=True)
class ToolDefinition:
    """Describe a tool and whether it can safely run without approval."""

    handler: Callable[[], dict[str, Any]]
    read_only: bool = True
    description: str = ""


def execute_tool(
    state: AgentState,
    tool_name: str,
    tool: Callable[[], dict[str, Any]] | ToolDefinition,
    approved: bool = False,
) -> dict[str, Any]:
    """Execute a tool while enforcing approval, call limits, and result checks."""

    if isinstance(tool, ToolDefinition):
        if not tool.read_only and not approved:
            return {
                "success": False,
                "blocked": True,
                "error": f"Human approval required for tool: {tool_name}",
            }
        handler = tool.handler
    else:
        handler = tool

    if state.tool_calls >= MAX_TOOL_CALLS:
        return {
            "success": False,
            "blocked": False,
            "error": "Maximum tool-call limit reached.",
        }

    state.tool_calls += 1

    try:
        result = handler()

        if not isinstance(result, dict):
            raise ValueError(
                "Invalid result from diagnostic tool: expected a dictionary."
            )

        state.actions_taken.append({
            "tool": tool_name,
            "status": "completed",
        })
        state.findings.append({
            "tool": tool_name,
            "result": result,
        })

        return {"success": True, "result": result}

    except Exception as exc:
        state.actions_taken.append({
            "tool": tool_name,
            "status": "failed",
            "error": str(exc),
        })
        return {"success": False, "error": str(exc)}
