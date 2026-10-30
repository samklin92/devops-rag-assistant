from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# 1. Define our workflow state
class IncidentState(TypedDict):
    incident: str
    finding: str
    status: str


# 2. First node: inspect the incident
def inspect_incident(state: IncidentState) -> dict:
    print("Inspecting Kubernetes incident...")

    return {
        "finding": "Pod events indicate an image authentication problem."
    }


# 3. Second node: record investigation status
def record_finding(state: IncidentState) -> dict:
    print("Recording investigation findings...")

    return {
        "status": "Investigation findings recorded."
    }


# 4. Create the graph
builder = StateGraph(IncidentState)


# 5. Register nodes
builder.add_node("inspect_incident", inspect_incident)
builder.add_node("record_finding", record_finding)


# 6. Connect nodes
builder.add_edge(START, "inspect_incident")
builder.add_edge("inspect_incident", "record_finding")
builder.add_edge("record_finding", END)


# 7. Create an in-memory checkpointer
checkpointer = InMemorySaver()


# 8. Compile graph with checkpointing enabled
graph = builder.compile(checkpointer=checkpointer)


# 9. Identify this investigation
config = {
    "configurable": {
        "thread_id": "kubernetes-incident-001"
    }
}


# 10. Execute workflow
result = graph.invoke(
    {
        "incident": "Production pod showing ImagePullBackOff",
        "finding": "",
        "status": "pending",
    },
    config=config,
)


print("\nFinal workflow result:")
print(result)


# 11. Retrieve the saved checkpoint
saved_state = graph.get_state(config)


print("\nRetrieved checkpoint:")
print(saved_state.values)


print("\nCheckpoint thread ID:")
print(saved_state.config["configurable"]["thread_id"])

# Start a second independent investigation
second_config = {
    "configurable": {
        "thread_id": "kubernetes-incident-002"
    }
}

second_result = graph.invoke(
    {
        "incident": "Production pod showing CrashLoopBackOff",
        "finding": "",
        "status": "pending",
    },
    config=second_config,
)

print("\nSecond investigation:")
print(graph.get_state(second_config).values)