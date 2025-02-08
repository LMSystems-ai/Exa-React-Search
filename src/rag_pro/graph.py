"""
This script creates a ReAct-style agent that uses an Exa search tool node.
The Exa search tool uses the Exa API (via the exa_py SDK) to search for webpages
while filtering by specific domains via the `include_domains` parameter.
"""

import json
import os

# Import Exa and LangChain/LangGraph tools
from exa_py import Exa
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent - simplified to focus on message passing"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    include_domains: list[str]

# Create Exa client with proper error handling
# exa_api_key = os.environ.get("EXA_API_KEY")
# if not exa_api_key:
#     raise ValueError("EXA_API_KEY environment variable must be set")
# print("EXA_API_KEY", exa_api_key)


exa = Exa(api_key=os.getenv("EXA_API_KEY"))

@tool
def exa_search(
    query: str,
    include_domains: list[str] | None = None,
) -> list[dict]:
    """Search for webpages using Exa's neural search API and retrieve contents with highlights.

    Args:
        query: The search query string
        include_domains: Optional list of domains to filter results by
    """
    search_params = {
        "use_autoprompt": True,
        "num_results": 3,
        "text": True,
        "highlights": True,
    }

    if include_domains:
        search_params["include_domains"] = include_domains

    results = exa.search_and_contents(
        query,
        **search_params
    ).results
    # Convert result objects into dictionaries so they can be JSON serialized
    return [r.__dict__ for r in results]

tools = [exa_search]
tools_by_name = {tool.name: tool for tool in tools}  # Add this mapping

# Initialize model with proper tool binding
llm = ChatOpenAI(model="o3-mini-2025-01-31", api_key=os.getenv("OPENAI_API_KEY"))
model = llm.bind_tools(tools)

# Define tool node with proper state handling
def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    responses = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] not in tools_by_name:
            raise ValueError(f"Tool {tool_call['name']} not found")

        tool = tools_by_name[tool_call["name"]]
        # Safely get the include_domains from state if it exists
        include_domains = state.get("include_domains")
        tool_args = {**tool_call["args"]}
        if include_domains is not None:
            tool_args["include_domains"] = include_domains
        output = tool.invoke(tool_args)
        responses.append(
            ToolMessage(
                content=json.dumps(output, indent=2),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": responses}

# Simplified system message setup
system_message = SystemMessage(
    content="You are a web research assistant. Use Exa search to find information and always cite sources with markdown hyperlinks."
)

def call_model(state: AgentState, config: RunnableConfig):
    # Prepend system message only once
    all_messages = [system_message] + state["messages"]
    response = model.invoke(all_messages, config)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Now we can compile and visualize our graph
graph = workflow.compile()