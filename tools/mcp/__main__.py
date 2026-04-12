"""Launch RASPUTIN MCP server.

Run from the repository root::

    python3 tools/mcp/server.py          # preferred
    python3 tools/mcp/__main__.py        # alternative
"""

from server import MCP_HOST, MCP_PORT, mcp  # same-directory import

mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
