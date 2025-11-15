import asyncio
from fastmcp import Client
from typing import List
import uuid

from ragprod.domain.document import Document

import asyncio
import uuid
from fastmcp import Client


async def main():
    client = Client("http://localhost:8000/mcp")

    async with client:
        # List tools to check your tools are available
        tools = await client.list_tools()
        print("Available tools:", [tool.name for tool in tools])

        # Create 4 example documents with original content
        docs: List[Document] = [
            Document(
                id=str(uuid.uuid4()),
                raw_text="The Eiffel Tower was inaugurated in 1889 and is one of the most visited landmarks in the world.",
                source="history",
                title="Eiffel Tower"
            ),
            Document(
                id=str(uuid.uuid4()),
                raw_text="Python is a high-level programming language known for its readability, created by Guido van Rossum in 1991.",
                source="programming",
                title="Python Language"
            ),
            Document(
                id=str(uuid.uuid4()),
                raw_text="The Amazon rainforest produces about 20% of the world’s oxygen and is home to millions of species.",
                source="nature",
                title="Amazon Rainforest"
            ),
            Document(
                id=str(uuid.uuid4()),
                raw_text="Light travels at approximately 299,792 kilometers per second in a vacuum, a fundamental constant in physics.",
                source="physics",
                title="Speed of Light"
            ),
        ]

        # Call add_documents tool
        print("Adding documents...")
        result_add = await client.call_tool("add_documents", {"documents": [d.dict() for d in docs]})
        print("add_documents result:", result_add.data if hasattr(result_add, "data") else result_add)

        # Now retrieve with rag_retrieve
        query = "Where was the Eiffel Tower built and when?"
        print("Retrieving documents for query:", query)
        result_retrieve = await client.call_tool("rag_retrieve", {"query": query, "limit": 3})
        print("rag_retrieve result:", result_retrieve.data if hasattr(result_retrieve, "data") else result_retrieve)

        # Try another question
        query2 = "How fast does light travel?"
        print("Retrieving documents for query:", query2)
        result_retrieve2 = await client.call_tool("rag_retrieve", {"query": query2, "limit": 3})
        print("rag_retrieve result 2:", result_retrieve2.data if hasattr(result_retrieve2, "data") else result_retrieve2)

if __name__ == "__main__":
    asyncio.run(main())
