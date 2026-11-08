from mcp.server import MCPServer

# Create our MCP server
mcp = MCPServer(
    name="devops-kubernetes-server",
    version="1.0.0",
    instructions="Provides read-only Kubernetes diagnostic tools."
)


# Simulated Kubernetes pod data
SIMULATED_PODS = {
    "payment-api": {
        "name": "payment-api",
        "namespace": "production",
        "status": "Running",
        "ready": "1/1",
        "restarts": 0,
        "node": "worker-node-01"
    },
    "inventory-api": {
        "name": "inventory-api",
        "namespace": "production",
        "status": "CrashLoopBackOff",
        "ready": "0/1",
        "restarts": 5,
        "node": "worker-node-02"
    }
}


# Register our first MCP tool
@mcp.tool(
    name="inspect_pod",
    description="Inspect a Kubernetes pod's status and details.",
    structured_output=True
)
def inspect_pod(pod_name: str) -> dict[str, str | int]:
    """Retrieve information about a Kubernetes pod."""

    pod = SIMULATED_PODS.get(pod_name)

    if pod is None:
        return {
            "error": f"Pod '{pod_name}' was not found."
        }

    return pod

# Simulated Kubernetes pod events
SIMULATED_POD_EVENTS = {
    "payment-api": [
        {
            "type": "Normal",
            "reason": "Scheduled",
            "message": "Successfully assigned payment-api to worker-node-01"
        }
    ],
    "inventory-api": [
        {
            "type": "Warning",
            "reason": "BackOff",
            "message": "Back-off restarting failed container"
        },
        {
            "type": "Normal",
            "reason": "Pulled",
            "message": "Container image successfully pulled"
        }
    ]
}


# Register our second MCP tool
@mcp.tool(
    name="get_pod_events",
    description="Retrieve Kubernetes events associated with a pod.",
    structured_output=True
)
def get_pod_events(pod_name: str) -> list[dict[str, str]]:
    """Retrieve diagnostic events for a Kubernetes pod."""

    return SIMULATED_POD_EVENTS.get(pod_name, [])
# Simulated Kubernetes container logs
SIMULATED_CONTAINER_LOGS = {
    "payment-api": [
        "Application started successfully.",
        "Connected to payment database.",
        "Health check passed."
    ],
    "inventory-api": [
        "Starting inventory service...",
        "Connecting to database...",
        "ERROR: Database connection refused.",
        "Application startup failed."
    ]
}


# Register our third MCP tool
@mcp.tool(
    name="get_container_logs",
    description="Retrieve container logs for a Kubernetes pod.",
    structured_output=True
)
def get_container_logs(pod_name: str) -> list[str]:
    """Retrieve container logs for a Kubernetes pod."""

    return SIMULATED_CONTAINER_LOGS.get(
        pod_name,
        ["No logs found for this pod."]
    )

# Simulated Kubernetes deployment data
SIMULATED_DEPLOYMENTS = {
    "payment-api": {
        "name": "payment-api",
        "namespace": "production",
        "desired_replicas": 3,
        "ready_replicas": 3,
        "available_replicas": 3,
        "status": "Healthy"
    },
    "inventory-api": {
        "name": "inventory-api",
        "namespace": "production",
        "desired_replicas": 3,
        "ready_replicas": 1,
        "available_replicas": 1,
        "status": "Degraded"
    }
}


# Register our fourth MCP tool
@mcp.tool(
    name="check_deployment_status",
    description="Check the health and replica status of a Kubernetes deployment.",
    structured_output=True
)
def check_deployment_status(deployment_name: str) -> dict[str, str | int]:
    """Retrieve Kubernetes deployment health information."""

    deployment = SIMULATED_DEPLOYMENTS.get(deployment_name)

    if deployment is None:
        return {
            "error": f"Deployment '{deployment_name}' was not found."
        }

    return deployment

if __name__ == "__main__":
    mcp.run(transport="stdio")