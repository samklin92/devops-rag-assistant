"""State management for DevOps agent investigations."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class AgentState:
    """Track the progress and evidence of a single agent task."""

    task: str
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "pending"
    findings: list[dict[str, Any]] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    knowledge: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    next_action: str | None = None
