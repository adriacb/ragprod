from ragprod.presentation.mcp.server import mcp
import asyncio

async def main():
    await mcp.run_async(transport="http", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    asyncio.run(main())