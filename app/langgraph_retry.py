from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class IncidentState(TypedDict):
    incident: str
    attempts: int
    max_attempts: int
    status: str
    error: str


def inspect_pod(state: IncidentState) -> dict:

    attempts = state["attempts"] + 1

    print(f"Inspecting Kubernetes pod. Attempt {attempts}")

    return {
        "attempts": attempts,
        "status": "retry",
        "error": "Kubernetes API temporarily unavailable."
    }


def decide_next_step(state: IncidentState) -> str:

    if state["status"] == "success":
        return "complete"

    if state["attempts"] >= state["max_attempts"]:
        return "stop"

    return "retry"


def complete(state: IncidentState) -> dict:

    print("Pod inspection completed successfully.")

    return {
        "status": "completed"
    }


def stop(state: IncidentState) -> dict:

    print("Maximum retry attempts reached.")

    return {
        "status": "failed"
    }


builder = StateGraph(IncidentState)

builder.add_node("inspect_pod", inspect_pod)
builder.add_node("complete", complete)
builder.add_node("stop", stop)

builder.add_edge(START, "inspect_pod")

builder.add_conditional_edges(
    "inspect_pod",
    decide_next_step,
    {
        "retry": "inspect_pod",
        "complete": "complete",
        "stop": "stop",
    },
)

builder.add_edge("complete", END)
builder.add_edge("stop", END)

graph = builder.compile()


initial_state = {
    "incident": "Production pod showing ImagePullBackOff",
    "attempts": 0,
    "max_attempts": 3,
    "status": "pending",
    "error": "",
}

result = graph.invoke(initial_state)

print("\nFinal result:")
print(result)