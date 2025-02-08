
Langgraph project which uses Exa and OpenAI to search the web and answer questions.

What would be really cool is allowing the llm to call the tool node (exa search) with multiple queries that then run in parallel.

### Quick Start

add your OpenAI and Exa API keys to the .env file

```bash
pip install -e .
```

```bash
pip install langgraph-cli[inmem]
```

```bash
langgraph dev
```