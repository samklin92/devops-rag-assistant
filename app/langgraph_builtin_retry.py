from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


class IncidentState(TypedDict):
    incident: str
    attempts: int
    status: str


inspection_attempts = 0


def inspect_pod(state: IncidentState) -> dict:

    global inspection_attempts
    inspection_attempts += 1

    print(f"Inspecting Kubernetes pod. Attempt {inspection_attempts}")

    # Simulate temporary API failures
    if inspection_attempts <= 3:
        raise ConnectionError("Kubernetes API temporarily unavailable.")

    return {
        "attempts": inspection_attempts,
        "status": "success"
    }


def complete(state: IncidentState) -> dict:

    print("Pod inspection completed successfully.")

    return {"status": "completed"}


builder = StateGraph(IncidentState)

builder.add_node(
    "inspect_pod",
    inspect_pod,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.1,
        backoff_factor=2.0,
        max_interval=1.0,
        jitter=False,
    ),
)

builder.add_node("complete", complete)

builder.add_edge(START, "inspect_pod")
builder.add_edge("inspect_pod", "complete")
builder.add_edge("complete", END)

graph = builder.compile()

initial_state = {
    "incident": "Production pod showing ImagePullBackOff",
    "attempts": 0,
    "status": "pending",
}

try:
    result = graph.invoke(initial_state)

    print("\nFinal result:")
    print(result)

except ConnectionError as error:
    print("\nInvestigation failed after retry attempts.")
    print(f"Reason: {error}")