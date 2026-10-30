"""LangGraph incident workflow with human approval and SQLite checkpoints.

Usage:
    python incident_workflow.py            # run until the approval pause
    python incident_workflow.py recover    # reload the checkpoint and resume
"""

import sys
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

CHECKPOINT_DB = "devops_agent_checkpoints.sqlite"
THREAD_ID = "production-incident-001"


class IncidentState(TypedDict):
    incident: str
    finding: str
    approval_status: str
    execution_status: str


# ------------------------------------------------------------------
# Workflow nodes
# ------------------------------------------------------------------


def investigate_incident(state: IncidentState) -> dict:
    print("Investigating Kubernetes incident...")

    return {
        "finding": "Pod events indicate an image authentication problem."
    }


def request_approval(state: IncidentState) -> dict:
    decision = interrupt(
        {
            "message": "Human approval required.",
            "incident": state["incident"],
            "finding": state["finding"],
            "question": "Do you approve the proposed remediation?",
        }
    )

    return {
        "approval_status": "approved" if decision is True else "rejected"
    }


def decide_after_approval(state: IncidentState) -> str:
    if state["approval_status"] == "approved":
        return "complete_operation"

    return "cancel_operation"


def complete_operation(state: IncidentState) -> dict:
    print("Approval received. Simulating remediation...")

    return {"execution_status": "Remediation completed successfully."}


def cancel_operation(state: IncidentState) -> dict:
    print("Approval rejected. Remediation cancelled.")

    return {"execution_status": "Remediation cancelled."}


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------


def build_graph(checkpointer):
    builder = StateGraph(IncidentState)

    # Register workflow nodes.
    builder.add_node("investigate_incident", investigate_incident)
    builder.add_node("request_approval", request_approval)
    builder.add_node("complete_operation", complete_operation)
    builder.add_node("cancel_operation", cancel_operation)

    # Define workflow connections.
    builder.add_edge(START, "investigate_incident")
    builder.add_edge("investigate_incident", "request_approval")

    # Route based on human approval.
    builder.add_conditional_edges(
        "request_approval",
        decide_after_approval,
        {
            "complete_operation": "complete_operation",
            "cancel_operation": "cancel_operation",
        },
    )

    builder.add_edge("complete_operation", END)
    builder.add_edge("cancel_operation", END)

    return builder.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


def save_checkpoint():
    """Run the workflow until it pauses for human approval."""
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        existing = graph.get_state(config)

        if existing.next:
            print(
                f"Thread '{THREAD_ID}' is already paused at: "
                f"{existing.next}"
            )
            print("Run 'python incident_workflow.py recover' to resume it,")
            print(f"or delete {CHECKPOINT_DB} to start over.")
            return

        initial_state = {
            "incident": "Production application experiencing instability",
            "finding": "",
            "approval_status": "pending",
            "execution_status": "waiting",
        }

        result = graph.invoke(initial_state, config=config)

        print("\nWorkflow paused.")
        print("Approval request:")
        print(result.get("__interrupt__"))

        saved_state = graph.get_state(config)

        print("\nCheckpoint saved.")
        print("Saved workflow state:")
        print(saved_state.values)


def recover_checkpoint():
    """Reload the saved checkpoint and resume the paused workflow."""
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        saved_state = graph.get_state(config)

        if not saved_state.values:
            print("No checkpoint found. Run the script without arguments first.")
            return

        print("\nRecovered checkpoint:")
        print(saved_state.values)

        if not saved_state.next:
            print("\nThis workflow has already finished. Nothing to resume.")
            return

        print("\nResuming investigation...")

        result = graph.invoke(Command(resume=True), config=config)

        print("\nRecovered workflow result:")
        print(result)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recover":
        recover_checkpoint()
    else:
        save_checkpoint()