from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define workflow state
class IncidentState(TypedDict):
    incident: str
    evidence_count: int
    investigation_attempts: int
    max_attempts: int
    status: str


# 2. Inspect the pod
def inspect_pod(state: IncidentState) -> dict:
    print("\nInspecting Kubernetes pod...")

    return {
        "investigation_attempts": state["investigation_attempts"] + 1
    }


# 3. Evaluate available evidence
def evaluate_evidence(state: IncidentState) -> dict:
    print("Evaluating available evidence...")

    print("Insufficient evidence.")
    return {"evidence_count": 1}


# 4. Decide what happens next
def decide_next_step(state: IncidentState) -> str:
    if state["evidence_count"] >= 3:
        return "generate_response"

    if state["investigation_attempts"] >= state["max_attempts"]:
        return "stop_investigation"

    return "gather_diagnostics"


# 5. Gather additional diagnostics
def gather_diagnostics(state: IncidentState) -> dict:
    print("Gathering additional Kubernetes diagnostics...")

    return {
        "status": "Additional diagnostics gathered"
    }


# 6. Generate final response
def generate_response(state: IncidentState) -> dict:
    print("Generating investigation report...")

    return {
        "status": "Investigation completed successfully"
    }


# 7. Handle investigation limits
def stop_investigation(state: IncidentState) -> dict:
    print("Maximum investigation attempts reached.")

    return {
        "status": "Investigation stopped: insufficient evidence"
    }


# 8. Create graph
builder = StateGraph(IncidentState)


# 9. Register nodes
builder.add_node("inspect_pod", inspect_pod)
builder.add_node("evaluate_evidence", evaluate_evidence)
builder.add_node("gather_diagnostics", gather_diagnostics)
builder.add_node("generate_response", generate_response)
builder.add_node("stop_investigation", stop_investigation)


# 10. Connect nodes
builder.add_edge(START, "inspect_pod")
builder.add_edge("inspect_pod", "evaluate_evidence")

builder.add_conditional_edges(
    "evaluate_evidence",
    decide_next_step,
    {
        "gather_diagnostics": "gather_diagnostics",
        "generate_response": "generate_response",
        "stop_investigation": "stop_investigation",
    },
)


# 11. Create the loop
builder.add_edge("gather_diagnostics", "inspect_pod")

# 12. Connect terminal nodes
builder.add_edge("generate_response", END)
builder.add_edge("stop_investigation", END)


# 13. Compile graph
graph = builder.compile()


# 14. Execute workflow
result = graph.invoke({
    "incident": "Kubernetes pod showing CrashLoopBackOff",
    "evidence_count": 0,
    "investigation_attempts": 0,
    "max_attempts": 3,
    "status": "pending",
})


print("\nFinal workflow state:")
print(result)