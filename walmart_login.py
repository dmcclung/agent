"""One-time Walmart login for the MCP server, run outside the bot so credentials never pass through chat.

The session is saved in ~/.striderlabs/walmart/ and reused by main.py.
"""
import asyncio
from getpass import getpass

from langchain_mcp_adapters.tools import load_mcp_tools

from main import mcp_client


async def run():
    email = input("Walmart email: ")
    password = getpass("Walmart password: ")
    async with mcp_client.session("walmart") as session:
        tools = {t.name: t for t in await load_mcp_tools(session)}
        print(await tools["login"].ainvoke({"email": email, "password": password, "headless": False}))
        print(await tools["status"].ainvoke({}))


if __name__ == "__main__":
    asyncio.run(run())
