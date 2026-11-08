"""Adapters connecting the DevOps agent to Kubernetes MCP tools."""

from typing import Any


def build_mcp_tools(
    session: Any,
    pod_name: str,
) -> dict:
    """Build agent-compatible asynchronous handlers from MCP tools."""

    async def inspect_pod_events() -> dict:
        """Retrieve Kubernetes events for the target pod."""

        response = await session.call_tool(
            "get_pod_events",
            arguments={"pod_name": pod_name},
        )

        if response.is_error:
            raise RuntimeError("MCP failed to retrieve pod events.")

        return response.structured_content

    async def gather_additional_diagnostics() -> dict:
        """Collect pod details, container logs and deployment health."""

        pod = await session.call_tool(
            "inspect_pod",
            arguments={"pod_name": pod_name},
        )

        logs = await session.call_tool(
            "get_container_logs",
            arguments={"pod_name": pod_name},
        )

        deployment = await session.call_tool(
            "check_deployment_status",
            arguments={"deployment_name": pod_name},
        )

        responses = [pod, logs, deployment]

        if any(response.is_error for response in responses):
            raise RuntimeError(
                "MCP failed to collect additional diagnostics."
            )

        return {
            "pod": pod.structured_content,
            "logs": logs.structured_content,
            "deployment": deployment.structured_content,
            "terminal": True,
        }

    return {
        "inspect_pod_events": inspect_pod_events,
        "gather_additional_diagnostics": gather_additional_diagnostics,
    }