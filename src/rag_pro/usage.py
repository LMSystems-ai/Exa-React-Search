
from graph import graph


# Helper function for formatting the stream nicely
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "what's the functional api and how is it different from the regular langgraph graph api?")]}
print_stream(graph.stream(inputs, stream_mode="values"))