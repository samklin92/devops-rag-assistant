import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # Tell the client how to start our MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "app.mcp_server.server"]
    )

    # Establish communication with the MCP server
    async with stdio_client(server_params) as (read_stream, write_stream):

        async with ClientSession(read_stream, write_stream) as session:

            # Initialize the MCP connection
            await session.initialize()

            print("\nConnected to Kubernetes MCP server!")

            # Discover available tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:")

            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # Call our first diagnostic tool
            result = await session.call_tool(
                "inspect_pod",
                arguments={"pod_name": "inventory-api"}
            )

            print("\nPod inspection result:")
            print(result.structured_content)

            # Retrieve pod events
            events = await session.call_tool(
            "get_pod_events",
            arguments={"pod_name": "inventory-api"}
      )

            print("\nPod events:")
            print(events.structured_content)

            # Retrieve container logs
            logs = await session.call_tool(
            "get_container_logs",
            arguments={"pod_name": "inventory-api"}
   )

            print("\nContainer logs:")
            print(logs.structured_content)

            # Check deployment health
            deployment = await session.call_tool(
            "check_deployment_status",
            arguments={"deployment_name": "inventory-api"}
   )

            print("\nDeployment status:")
            print(deployment.structured_content)

if __name__ == "__main__":
    asyncio.run(main())