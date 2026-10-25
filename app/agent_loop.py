"""Controlled investigation loop for the DevOps agent."""

from typing import Any, Callable

from app.agent_decision import decide_next_action
from app.agent_knowledge import retrieve_agent_knowledge
from app.agent_response import generate_agent_response
from app.agent_state import AgentState
from app.tool_executor import MAX_TOOL_CALLS, ToolDefinition, execute_tool


def recommend_next_step(
    status: str,
    reason: str,
    state: AgentState,
) -> str:
    """Recommend a practical next step when an investigation cannot continue."""

    if any(
        item.get("status") == "failed"
        for item in state.actions_taken
    ):
        return (
            "Check the failed diagnostic tool, its permissions, and its "
            "dependencies. Resolve the tool error before retrying."
        )

    if status == "blocked":
        if not state.knowledge:
            return (
                f"{reason} No relevant RAG knowledge is available. "
                "Register the required diagnostic tool or choose an "
                "available alternative, consult relevant DevOps "
                "documentation, and escalate if the issue remains unresolved."
            )

        sources = sorted({
            item.get("source", "unknown source")
            for item in state.knowledge
        })
        source_list = ", ".join(sources)

        return (
            f"{reason} Register the required diagnostic tool or choose "
            "an available alternative. Review the collected findings "
            f"alongside the retrieved guidance in {source_list}. "
            "If the issue remains unresolved, escalate to a human operator."
        )

    if not state.knowledge:
        return (
            "No relevant RAG knowledge is available. Consult relevant "
            "DevOps documentation, review the collected findings, and "
            "escalate to a human operator if the issue remains unresolved."
        )

    sources = sorted({
        item.get("source", "unknown source")
        for item in state.knowledge
    })
    source_list = ", ".join(sources)

    return (
        "Review the collected findings alongside the retrieved guidance "
        f"in {source_list}. Identify the missing evidence and choose a "
        "targeted diagnostic. If the issue remains unresolved, escalate "
        "to a human operator."
    )


def run_agent(
    state: AgentState,
    tools: dict[str, Callable[[], dict[str, Any]]],
    max_steps: int = MAX_TOOL_CALLS,
    embedding_model: Any = None,
    client: Any = None,
    approved_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Run a controlled investigation and generate a final response."""

    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer.")

    # The controller must never exceed the executor's tool-call safety limit.
    step_limit = min(max_steps, MAX_TOOL_CALLS)

    state.status = "running"
    steps = 0

    if embedding_model is not None:
        try:
            state.knowledge = retrieve_agent_knowledge(
                question=state.task,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            state.status = "failed"
            return {
                "status": state.status,
                "reason": f"Knowledge retrieval failed: {exc}",
                "next_step": (
                    "Check the retrieval error and verify the embedding "
                    "model, vector index, and knowledge-base configuration."
                ),
                "steps": steps,
            }

    for _ in range(step_limit):
        action = decide_next_action(state)

        if action not in tools:
            state.status = "blocked"
            reason = f"Tool is not registered: {action}"
            return {
                "status": state.status,
                "reason": reason,
                "next_step": recommend_next_step(
                    state.status, reason, state
                ),
                "steps": steps,
            }

        if any(
            item.get("tool") == action
            for item in state.actions_taken
        ):
            state.status = "needs_review"
            reason = f"Repeated action detected: {action}"
            return {
                "status": state.status,
                "reason": reason,
                "next_step": recommend_next_step(
                    state.status, reason, state
                ),
                "steps": steps,
            }

        tool = tools[action]
        approved = approved_actions is not None and action in approved_actions

        # Block infrastructure-changing tools unless explicitly approved.
        if (
            isinstance(tool, ToolDefinition)
            and not tool.read_only
            and not approved
        ):
            state.status = "blocked"
            reason = f"Human approval required for tool: {action}"
            return {
                "status": state.status,
                "reason": reason,
                "next_step": (
                    "Review the proposed infrastructure-changing action, "
                    "assess its impact, and obtain explicit human approval "
                    "before retrying the investigation."
                ),
                "steps": steps,
            }

        response = execute_tool(
            state,
            action,
            tool,
            approved=approved,
        )
        steps += 1

        if not response["success"]:
            state.status = "blocked" if response.get("blocked") else "failed"
            return {
                "status": state.status,
                "reason": response["error"],
                "next_step": recommend_next_step(
                    state.status, response["error"], state
                ),
                "steps": steps,
            }

        result = response["result"]

        if result.get("terminal") is True:
            state.status = "completed"
            state.next_action = None

            final_response = None

            if client is not None:
                try:
                    final_response = generate_agent_response(
                        state=state,
                        client=client,
                    )
                except Exception as exc:
                    state.status = "needs_review"
                    reason = (
                        "Investigation finished, but final response "
                        f"generation failed: {exc}"
                    )
                    return {
                        "status": state.status,
                        "reason": reason,
                        "next_step": (
                            "Check the response-generation error and verify "
                            "the LLM client, API configuration, and service "
                            "availability. Retry response generation using "
                            "the collected findings and retrieved knowledge."
                        ),
                        "steps": steps,
                        "final_response": None,
                    }

            return {
                "status": state.status,
                "steps": steps,
                "final_response": final_response,
            }

    state.status = "needs_review"
    reason = "Maximum investigation steps reached."

    return {
        "status": state.status,
        "reason": reason,
        "next_step": recommend_next_step(
            state.status, reason, state
        ),
        "steps": steps,
    }
