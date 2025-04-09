
# Exa-React-Search

A LangGraph project which uses Exa and OpenAI to search the web and answer questions.

What would be really cool is allowing the LLM to call the tool node (exa search) with multiple queries that then run in parallel.

## Quick Start (Local Development)

1. Add your OpenAI and Exa API keys to the `.env` file (see `.env.example` for required variables)

```bash
# Copy the example env file and add your API keys
cp .env.example .env
```

2. Install the package locally

```bash
pip install -e .
```

3. Run the development server

```bash
langgraph dev
```

## Docker Compose Setup

This project can be run using Docker Compose, which simplifies deployment and ensures consistent environments.

### Prerequisites

- Docker and Docker Compose installed on your system
- OpenAI and Exa API keys

### Setup Instructions

1. Create a `.env` file with your API keys:

```bash
# Copy the example env file
cp .env.example .env

# Edit the file to add your API keys
EXA_API_KEY=your_exa_api_key
OPENAI_API_KEY=your_openai_api_key
```

2. Build and start the Docker containers:

```bash
# Build the containers
docker compose build

# Start the services in detached mode
docker compose up -d
```

3. Verify the service is running:

```bash
# Check if the container is running
docker compose ps

# Check the logs
docker compose logs
```

4. To stop the service:

```bash
docker compose down
```

### Rebuilding After Changes

If you make changes to the code, you'll need to rebuild the Docker image:

```bash
# Stop the current containers
docker compose down

# Rebuild without using cache
docker compose build --no-cache

# Start the services again
docker compose up -d
```

## API Documentation

The API runs on port 2024 by default. Before using the API, ensure you have:
- Valid OpenAI API key with sufficient quota
- Valid Exa API key

The following endpoints are available:

### API Documentation Endpoints

- `GET /docs` - Interactive API documentation
- `GET /openapi.json` - OpenAPI specification

### Assistant Endpoints

- `POST /assistants/search` - Create a new search assistant
  - Request body: `{}`
  - Returns: Array of available assistants with their IDs and configurations
  - Example response:
    ```json
    [
      {
        "assistant_id": "76618b04-4f13-5cc0-ab0c-4989fdb8f806",
        "graph_id": "rag_pro",
        "config": {},
        "metadata": {"created_by": "system"},
        "name": "rag_pro",
        "created_at": "2025-04-09T15:49:14.998428+00:00",
        "updated_at": "2025-04-09T15:49:14.998428+00:00",
        "version": 1
      }
    ]
    ```

### Thread Management

- `POST /threads` - Create a new conversation thread
  - Request body: `{"metadata": {"source": "string"}}`
  - Returns: Thread ID, status, and metadata
  - Example response:
    ```json
    {
      "thread_id": "dfa1de85-f9bb-4ef0-bd33-86e318ea53b0",
      "created_at": "2025-04-09T18:57:43.854957+00:00",
      "updated_at": "2025-04-09T18:57:43.854962+00:00",
      "metadata": {"source": "test"},
      "status": "idle",
      "config": {}
    }
    ```

### Run Management

- `POST /threads/{thread_id}/runs` - Start a new run in a thread
  - Request body: 
    ```json
    {
      "assistant_id": "string",
      "input": {
        "messages": [
          {
            "role": "user",
            "content": "string"
          }
        ]
      }
    }
    ```
  - Returns: Run ID, status, and configuration
  - Example response:
    ```json
    {
      "run_id": "1f01574a-0654-65b4-9730-35371562741a",
      "thread_id": "dfa1de85-f9bb-4ef0-bd33-86e318ea53b0",
      "assistant_id": "76618b04-4f13-5cc0-ab0c-4989fdb8f806",
      "status": "pending",
      "metadata": {
        "created_by": "system",
        "source": "test",
        "graph_id": "rag_pro"
      }
    }
    ```
  - Possible status values: "pending", "in_progress", "completed", "error"

- `GET /threads/{thread_id}/runs/{run_id}` - Get the status of a specific run
  - Returns: Detailed run status and results
  - Example response:
    ```json
    {
      "run_id": "1f01574a-0654-65b4-9730-35371562741a",
      "thread_id": "dfa1de85-f9bb-4ef0-bd33-86e318ea53b0",
      "assistant_id": "76618b04-4f13-5cc0-ab0c-4989fdb8f806",
      "status": "completed",
      "metadata": {
        "created_by": "system",
        "source": "test",
        "graph_id": "rag_pro"
      }
    }
    ```

### Thread State

- `GET /threads/{thread_id}/state` - Get the current state of a thread
  - Returns: Current thread state including messages, tasks, and metadata
  - Example response:
    ```json
    {
      "values": {
        "messages": [
          {
            "content": "What is LangGraph?",
            "type": "human",
            "id": "55e4595d-386b-4350-8225-269501e02fae"
          }
        ]
      },
      "metadata": {
        "source": "loop",
        "graph_id": "rag_pro"
      }
    }
    ```

### Error Handling

The API may return various error responses:
- 400: Bad Request - Invalid input parameters
- 404: Not Found - Resource doesn't exist
- 429: Rate Limit - OpenAI API quota exceeded
- 500: Internal Server Error - Server-side issues

Always check the response status code and handle errors appropriately in your applications.

## Example Usage

Before testing the API endpoints, ensure you have:
1. The service running (either locally or via Docker)
2. Valid API keys in your `.env` file (OpenAI API key with sufficient quota and Exa API key)
3. The container name matches your environment (default is `exa-react-search-langgraph-app-1`)

You can interact with the API using curl commands. Here are examples for both direct access and through Docker:

### Direct Access

```bash
# Get API documentation
curl http://127.0.0.1:2024/docs

# Get OpenAPI specification
curl http://127.0.0.1:2024/openapi.json

# Create a search assistant
curl http://127.0.0.1:2024/assistants/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{}'

# Create a new thread
curl http://127.0.0.1:2024/threads \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"source": "test"}}'

# Create a new run in a thread (replace thread_id and assistant_id with actual values)
curl http://127.0.0.1:2024/threads/{thread_id}/runs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "{assistant_id}",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is LangGraph?"
        }
      ]
    }
  }'

# Get run status (replace thread_id and run_id with actual values)
curl http://127.0.0.1:2024/threads/{thread_id}/runs/{run_id}

# Get thread state (replace thread_id with actual value)
curl http://127.0.0.1:2024/threads/{thread_id}/state
```

### Through Docker Container

```bash
# Get API documentation
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/docs

# Get OpenAPI specification
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/openapi.json

# Create a search assistant
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/assistants/search \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{}'

# Create a new thread
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/threads \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"source": "test"}}'

# Create a new run in a thread (replace thread_id and assistant_id with actual values)
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/threads/{thread_id}/runs \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "{assistant_id}",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is LangGraph?"
        }
      ]
    }
  }'

# Get run status (replace thread_id and run_id with actual values)
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/threads/{thread_id}/runs/{run_id}

# Get thread state (replace thread_id with actual value)
docker exec -it exa-react-search-langgraph-app-1 curl http://127.0.0.1:2024/threads/{thread_id}/state
```

Note: Replace `{thread_id}`, `{run_id}`, and `{assistant_id}` with actual values obtained from previous API responses.
