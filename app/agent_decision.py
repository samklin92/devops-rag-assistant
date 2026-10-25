"""Rule-based decision-making for DevOps investigations."""

from app.agent_state import AgentState


def decide_next_action(state: AgentState) -> str:
    """Choose the next diagnostic action based on collected evidence."""

    if not state.findings:
        state.next_action = "inspect_pod_events"
        return state.next_action

    latest_finding = state.findings[-1]
    result = latest_finding.get("result", {})
    reason = result.get("reason", "")

    # Concrete diagnostic evidence takes priority.
    if reason == "FailedToRetrieveImagePullSecret":
        state.next_action = "inspect_image_pull_secret_configuration"

    elif reason == "ImageNotFound":
        state.next_action = "inspect_image_name_and_tag"

    # General RAG guidance alone does not establish the root cause.
    else:
        state.next_action = "gather_additional_diagnostics"

    return state.next_action