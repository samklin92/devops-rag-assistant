from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# 1. Define workflow state
class IncidentState(TypedDict):
    incident: str
    proposed_action: str
    approval_status: str
    execution_status: str


# 2. Investigate the incident
def investigate_incident(state: IncidentState) -> dict:
    print("Investigating Kubernetes incident...")

    return {
        "proposed_action": "Restart production deployment"
    }


# 3. Request human approval
def request_approval(state: IncidentState) -> dict:

    decision = interrupt({
        "message": "Human approval required.",
        "incident": state["incident"],
        "proposed_action": state["proposed_action"],
        "question": "Do you approve this operation?"
    })

    return {
        "approval_status": "approved" if decision is True else "rejected"
    }


# 4. Decide what happens after approval
def decide_after_approval(state: IncidentState) -> str:

    if state["approval_status"] == "approved":
        return "restart_deployment"

    return "stop_operation"


# 5. Execute the approved operation
def restart_deployment(state: IncidentState) -> dict:
    print("Restarting Kubernetes deployment...")

    return {
        "execution_status": "Deployment restart completed."
    }


# 6. Stop if approval is rejected
def stop_operation(state: IncidentState) -> dict:
    print("Operation cancelled. Human approval was not granted.")

    return {
        "execution_status": "Operation cancelled."
    }


# 7. Create graph
builder = StateGraph(IncidentState)


# 8. Register nodes
builder.add_node("investigate_incident", investigate_incident)
builder.add_node("request_approval", request_approval)
builder.add_node("restart_deployment", restart_deployment)
builder.add_node("stop_operation", stop_operation)


# 9. Connect nodes
builder.add_edge(START, "investigate_incident")
builder.add_edge("investigate_incident", "request_approval")

builder.add_conditional_edges(
    "request_approval",
    decide_after_approval,
    {
        "restart_deployment": "restart_deployment",
        "stop_operation": "stop_operation",
    },
)

builder.add_edge("restart_deployment", END)
builder.add_edge("stop_operation", END)


# 10. Configure checkpointing
checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)


# 11. Identify this investigation
config = {
    "configurable": {
        "thread_id": "production-incident-001"
    }
}


# 12. Start investigation
initial_result = graph.invoke(
    {
        "incident": "Production application experiencing instability",
        "proposed_action": "",
        "approval_status": "pending",
        "execution_status": "waiting",
    },
    config=config,
)


print("\nApproval request:")
print(initial_result)


# 13. Simulate human approval
print("\nHuman rejects the operation.")


# 14. Resume the paused workflow
final_result = graph.invoke(
    Command(resume=False),
    config=config,
)


print("\nFinal workflow state:")
print(final_result)