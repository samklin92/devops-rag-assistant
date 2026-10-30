from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. Define the information our workflow will carry
class IncidentState(TypedDict):
    incident: str
    investigation: str


# 2. First node: inspect the incident
def inspect_incident(state: IncidentState) -> dict:
    return {
        "investigation": f"Received incident: {state['incident']}"
    }


# 3. Second node: analyse the incident
def analyse_incident(state: IncidentState) -> dict:
    return {
        "investigation": (
            state["investigation"]
            + " | Analysis: Further diagnostic checks are required."
        )
    }


# 4. Create the graph
builder = StateGraph(IncidentState)


# 5. Register the nodes
builder.add_node("inspect_incident", inspect_incident)
builder.add_node("analyse_incident", analyse_incident)


# 6. Connect the nodes
builder.add_edge(START, "inspect_incident")
builder.add_edge("inspect_incident", "analyse_incident")
builder.add_edge("analyse_incident", END)


# 7. Compile the graph
graph = builder.compile()


# 8. Execute the workflow
result = graph.invoke({
    "incident": "Kubernetes pod is showing CrashLoopBackOff",
    "investigation": ""
})


print(result)