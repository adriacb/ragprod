from ragprod.presentation.mcp.server import mcp
from ragprod.presentation.mcp.tools.rag import rag_retrieve
import asyncio

async def main():
    await mcp.run_async(transport="http", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    asyncio.run(main())