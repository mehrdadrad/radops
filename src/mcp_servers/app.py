from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EchoServer")

@mcp.tool()
async def mcp_echo_tool(arg: str) -> str:
    """
    A simple echo tool for testing MCP connectivity and functionality.
    It returns the input argument prefixed with 'mcp echo: '.
    """
    return f"mcp echo: {arg}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")