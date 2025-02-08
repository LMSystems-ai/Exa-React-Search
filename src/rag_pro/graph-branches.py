import json
import os
import re

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
    """The state of the agent – simplified to focus on message passing"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create Exa client
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

@tool
def exa_search(
    query: str,
    include_domains: list[str] = ["langchain.com", "langchain-ai.github.io"],
) -> list[dict]:
    """Search for webpages using Exa's neural search API and retrieve contents with highlights."""
    results = exa.search_and_contents(
        query,
        use_autoprompt=True,
        num_results=3,
        include_domains=include_domains,
        text=True,
        highlights=True,
    ).results
    # Convert result objects into dictionaries so they can be JSON serialized
    return [r.__dict__ for r in results]

tools = [exa_search]
tools_by_name = {tool.name: tool for tool in tools}  # Mapping for easy lookup

llm = ChatOpenAI(model="o3-mini-2025-01-31", openai_api_key=os.getenv("OPENAI_API_KEY"))
model = llm.bind_tools(tools)

# We define a new helper that checks the tool output for XML tags and if found, spawns
# multiple "branches" to run exa_search calls in parallel (conceptually) for each query.
def process_agent_tool_output(tool_output: str) -> list[ToolMessage]:
    """
    This function takes a tool output string (assumed to be the agent’s text output) and checks
    for XML tags (e.g. <search>query here</search>). For each found query, it calls exa_search,
    and returns a list of tool messages containing the search results.

    If no XML tags are found, it simply returns a single ToolMessage with the original output.

    In a true parallel branch implementation using LangGraph, you might spawn separate branches.
    Here we simulate parallel branches by iterating over all queries.
    """
    # Use regex to find all queries inside <search> ... </search> tags
    queries = re.findall(r"<search>(.*?)</search>", tool_output, re.DOTALL)
    messages = []
    if queries:
        # For each search query, call exa_search and create a ToolMessage.
        for query in queries:
            # Trim whitespace from the query string
            query = query.strip()
            # Call the tool
            result = exa_search.invoke({"query": query})
            # Create a ToolMessage with the search result
            msg = ToolMessage(
                content=json.dumps(result, indent=2),
                name="exa_search",
                tool_call_id=f"exa_search_{query}",
            )
            messages.append(msg)
    else:
        # If no XML search tags are found, simply include the original output.
        messages.append(
            ToolMessage(
                content=tool_output,
                name="exa_search",
                tool_call_id="exa_search_default",
            )
        )
    return messages

# Modified tool node that checks for XML tags in the agent’s output and, if found, launches “parallel” calls.
def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    responses = []

    # Process each tool call from the last message
    for tool_call in last_message.tool_calls:
        if tool_call["name"] not in tools_by_name:
            raise ValueError(f"Tool {tool_call['name']} not found")

        # Instead of directly calling the tool, we first check the tool output
        # If the agent's message content contains XML-like search queries, we use those.
        tool_call_output = tool_call.get("output")
        if tool_call_output and "<search>" in tool_call_output:
            # Process the agent output to find XML tags and get messages for each branch.
            branch_messages = process_agent_tool_output(tool_call_output)
            responses.extend(branch_messages)
        else:
            # Fallback: use the tool as before
            tool = tools_by_name[tool_call["name"]]
            output = tool.invoke(tool_call["args"])
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
    content="You are a web research assistant. Use Exa search to find information and always cite sources."
)



def call_model(state: AgentState, config: RunnableConfig):
    # Prepend system message only once
    all_messages = [system_message] + state["messages"]
    response = model.invoke(all_messages, config)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish; otherwise, we continue.
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# Build the state graph
workflow = StateGraph(AgentState)

# Define the two nodes we cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# Add the conditional edge from agent to either tools or END.
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)

# Always transition from tools back to agent.
workflow.add_edge("tools", "agent")

# Finally, compile the graph.
graph = workflow.compile()

# Explanation:
# In this modified version:
# 1. The helper function process_agent_tool_output() uses regex to extract queries enclosed
#    in <search>...</search> tags.
# 2. In the tool_node() function, if the agent’s output (from tool_call["output"])
#    contains XML search tags, it spawns one exa_search call per query.
# 3. The responses (one per query) are then returned as separate ToolMessages, which in a
#    full LangGraph implementation could be seen as parallel branches.
# 4. For more details on branching, see the LangGraph How-To Guides on creating branches
#    for parallel execution.
#
# Make sure that when using this approach, your agent outputs XML wrapped queries for searching.
# Also note that while this example runs branch calls sequentially, LangGraph supports branching
# patterns (see https://langchain-ai.github.io/langgraph/how-tos/ for more details on parallel branches).

if __name__ == "__main__":
    # You can now run your graph with a starting message.
    init_state: AgentState = {"messages": []}
    # For example, you would push an initial message into your state.
    # Then use workflow.run(init_state) or your preferred method to execute the graph.
    pass