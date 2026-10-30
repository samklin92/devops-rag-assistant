from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define our workflow state
class IncidentState(TypedDict):
    incident: str
    diagnostic_reason: str
    next_step: str


# 2. Inspect the incident
def inspect_pod(state: IncidentState) -> dict:
    print("Inspecting Kubernetes pod...")

    return {
        "next_step": "Evaluating diagnostic evidence"
    }


# 3. Route based on diagnostic evidence
def decide_investigation(state: IncidentState) -> str:
    reason = state["diagnostic_reason"]

    if reason == "FailedToRetrieveImagePullSecret":
        return "inspect_secret"

    elif reason == "ImageNotFound":
        return "inspect_image"

    else:
        return "gather_diagnostics"


# 4. Investigate image pull secret
def inspect_secret(state: IncidentState) -> dict:
    print("Investigating image pull secret configuration.")

    return {
        "next_step": "Image pull secret investigation completed"
    }


# 5. Investigate image name and tag
def inspect_image(state: IncidentState) -> dict:
    print("Investigating image name and tag.")

    return {
        "next_step": "Image name and tag investigation completed"
    }


# 6. Gather additional evidence
def gather_diagnostics(state: IncidentState) -> dict:
    print("Gathering additional Kubernetes diagnostics.")

    return {
        "next_step": "Additional diagnostics gathered"
    }


# 7. Build the graph
builder = StateGraph(IncidentState)


# 8. Register nodes
builder.add_node("inspect_pod", inspect_pod)
builder.add_node("inspect_secret", inspect_secret)
builder.add_node("inspect_image", inspect_image)
builder.add_node("gather_diagnostics", gather_diagnostics)


# 9. Connect the nodes
builder.add_edge(START, "inspect_pod")

builder.add_conditional_edges(
    "inspect_pod",
    decide_investigation,
    {
        "inspect_secret": "inspect_secret",
        "inspect_image": "inspect_image",
        "gather_diagnostics": "gather_diagnostics",
    },
)


# 10. Connect investigation outcomes to END
builder.add_edge("inspect_secret", END)
builder.add_edge("inspect_image", END)
builder.add_edge("gather_diagnostics", END)


# 11. Compile the graph
graph = builder.compile()


# 12. Execute the workflow
result = graph.invoke({
    "incident": "Kubernetes pod showing ImagePullBackOff",
    "diagnostic_reason": "ImageNotFound",
    "next_step": "",
})


print("\nFinal workflow state:")
print(result)