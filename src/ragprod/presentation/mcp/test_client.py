from fastmcp.client.transports import StdioTransport


import os
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
import logging
from fastmcp import Client
from fastmcp.client.logging import LogMessage

# In a real app, you might configure this in your main entry point
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get a logger for the module where the client is used
logger = logging.getLogger(__name__)

# This mapping is useful for converting MCP level strings to Python's levels
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()

async def log_handler(message: LogMessage):
    """
    Handles incoming logs from the MCP server and forwards them
    to the standard Python logging system.
    """
    msg = message.data.get('msg')
    extra = message.data.get('extra')

    # Convert the MCP log level to a Python log level
    level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)

    # Log the message using the standard logging library
    logger.log(level, msg, extra=extra)

async def main():
    transport = StdioTransport(
        command="python",
        args=["src/ragprod/presentation/mcp/run.py"],
        # fil
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    client = Client(
        transport=transport,
        log_handler=log_handler
        )

    async with client:
        result = await client.call_tool(
            "rag_retrieve",
            {"query": "Machine learning", "limit": 5}
        )
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
