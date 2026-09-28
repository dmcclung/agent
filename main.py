import asyncio
import itertools
import json
import logging
import os
from typing import Annotated, TypedDict

from ddgs import DDGS

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

SIGNAL_HOST = "127.0.0.1"
SIGNAL_PORT = 7583
AGENT_NUMBER = os.environ["AGENT_NUMBER"]
ALLOWED_SENDERS = {s.strip() for s in os.environ["ALLOWED_SENDERS"].split(",") if s.strip()}
ALLOW_NOTE_TO_SELF = False
MODEL = os.environ["MODEL"]
SYSTEM_PROMPT = os.environ["SYSTEM_PROMPT"]
# Allowlist, not denylist: any tool the server adds later stays off until listed here.
# login/logout are excluded so credentials never pass through chat; log in once outside the bot.
WALMART_TOOLS = {
    "status", "set_address", "search", "get_product", "add_to_cart",
    "view_cart", "update_cart", "remove_from_cart", "checkout", "get_orders",
}

mcp_client = MultiServerMCPClient({
    "walmart": {
        "command": "npx",
        "args": ["-y", "@striderlabs/mcp-walmart"],
        "transport": "stdio",
    }
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("signal-agent")


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Input should be a search query."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return "\n\n".join(
            f"{r['title']} ({r['href']})\n{r['body']}" for r in results
        )


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(tools):
    llm_with_tools = ChatOllama(model=MODEL).bind_tools(tools)

    async def assistant(state: AgentState):
        response = await llm_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)

    return builder.compile(checkpointer=InMemorySaver())


class SignalClient:
    def __init__(self, host: str, port: int, account: str):
        self.host, self.port, self.account = host, port, account
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port, limit=2 ** 24
        )

    async def call(self, method: str, **params):
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": method,
            "params": {"account": self.account, **params},  # required in multi-account mode
        }
        async with self._write_lock:
            self.writer.write((json.dumps(payload) + "\n").encode())
            await self.writer.drain()

    async def send(self, recipient: str, text: str):
        await self.call("send", recipient=[recipient], message=text)

    async def typing(self, recipient: str, stop: bool = False):
        await self.call("sendTyping", recipient=[recipient], stop=stop)

    async def notifications(self):
        """Yield the params of every incoming 'receive' notification."""
        while True:
            line = await self.reader.readline()
            if not line:
                raise ConnectionError("signal-cli closed the connection")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in msg:
                log.warning("signal-cli error: %s", msg["error"])
            elif msg.get("method") == "receive":
                yield msg.get("params", {})


def extract_incoming(params: dict) -> tuple[str, str] | None:
    """Return (reply_to, text) for a text message we should answer, else None."""
    env = params.get("envelope", {})
    sender = env.get("sourceNumber") or env.get("source")

    # Normal message from another Signal account
    data = env.get("dataMessage") or {}
    if data.get("message") and not data.get("groupInfo"):  # ignore groups for now
        return sender, data["message"]

    # "Note to Self" typed on one of the bot account's other linked devices
    sent = (env.get("syncMessage") or {}).get("sentMessage") or {}
    if ALLOW_NOTE_TO_SELF and sent.get("message"):
        dest = sent.get("destinationNumber") or sent.get("destination")
        if sender == AGENT_NUMBER and dest == AGENT_NUMBER:
            return AGENT_NUMBER, sent["message"]

    return None


async def worker(client: SignalClient, queue: asyncio.Queue, graph):
    """Process messages one at a time so the GPU handles a single generation."""
    while True:
        reply_to, text = await queue.get()
        note_to_self = reply_to == AGENT_NUMBER
        try:
            if not note_to_self:
                await client.typing(reply_to)
            result = await graph.ainvoke(
                {"messages": [HumanMessage(text)]},
                config={"configurable": {"thread_id": reply_to}},
            )
            reply = result["messages"][-1].content or "(empty response)"
            await client.send(reply_to, reply)
            log.info("Replied to %s (%d chars)", reply_to, len(reply))
        except Exception:
            log.exception("Failed to handle message from %s", reply_to)
            try:
                await client.send(reply_to, "Sorry, something went wrong on my end.")
            except Exception:
                pass
        finally:
            queue.task_done()


async def main():
    # One long-lived session so the server's browser isn't relaunched on every tool call
    async with mcp_client.session("walmart") as session:
        available = await load_mcp_tools(session)
        walmart_tools = [t for t in available if t.name in WALMART_TOOLS]
        log.info("Walmart tools enabled: %s", sorted(t.name for t in walmart_tools))
        log.info("Walmart tools excluded: %s", sorted(t.name for t in available if t.name not in WALMART_TOOLS))
        graph = build_graph([web_search, *walmart_tools])

        client = SignalClient(SIGNAL_HOST, SIGNAL_PORT, AGENT_NUMBER)
        queue: asyncio.Queue = asyncio.Queue()
        asyncio.create_task(worker(client, queue, graph))

        while True:
            try:
                await client.connect()
                log.info("Connected to signal-cli at %s:%s", SIGNAL_HOST, SIGNAL_PORT)
                async for params in client.notifications():
                    if params.get("account") not in (None, AGENT_NUMBER):
                        continue
                    incoming = extract_incoming(params)
                    if not incoming:
                        continue
                    reply_to, text = incoming
                    if reply_to != AGENT_NUMBER and reply_to not in ALLOWED_SENDERS:
                        log.info("Ignoring message from non-allowlisted %s", reply_to)
                        continue
                    log.info("Message from %s: %s", reply_to, text[:80])
                    await queue.put(incoming)
            except (ConnectionError, OSError) as e:
                log.warning("Signal connection lost (%s); retrying in 5s", e)
                await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Shutting down")