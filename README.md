# Hybrid Graph RAG System

An advanced Retrieval-Augmented Generation (RAG) engine that intelligently routes queries across both structured knowledge graphs (Neo4j) and unstructured document stores. Built for enterprise scale, this system combines stateful agents (LangGraph), dynamic context retrieval, caching layers, and real-time token streaming to deliver highly accurate, low-latency insights.

## Core Architecture & Features

This system goes beyond basic vector search by orchestrating a dynamic multi-agent workflow:

- **Stateful Routing (LangGraph)**: Input queries are intelligently routed through a discrete state machine. The router determines whether a question requires structured data aggregation (Text-to-Cypher) or unstructured document retrieval.
- **Dynamic Few-Shot Prompt Tuning**: For structured queries, the engine performs a fast vector search against a repository of verified Cypher examples. Only the most semantically relevant schemas and examples are injected into the LLM context window, maximizing generation accuracy while minimizing token usage.
- **Strict Pre-Execution Security Guards**: All LLM-generated Cypher queries pass through a deterministic security layer before database execution. Mutating commands (`CREATE`, `MERGE`, `DELETE`) are outright rejected, relationship directionality is validated against strict schema triplets, and a `LIMIT 200` safeguard is enforced to prevent memory exhaustion.
- **Transaction Isolation**: Database connections are strictly isolated using `neo4j.AccessMode.READ`, guaranteeing zero accidental writes from rogue LLM generations.
- **Sub-Millisecond Redis Caching**: Natural language queries are normalized (case, whitespace, punctuation stripping) and hashed against a Redis instance. Repeat queries retrieve the exact, validated Cypher statements in sub-millisecond time, completely bypassing LLM inference latency.
- **Real-Time Token Streaming (SSE)**: The FastAPI backend streams the final response payload via Server-Sent Events (SSE). The `web_ui` consumes these events, resulting in a sub-second Time-to-First-Token (TTFT) experience.
- **Offline & Local Deployment**: The `LLM_PROVIDER` infrastructure natively supports hot-swapping between cloud providers (OpenAI) and local containerized models (Ollama), enabling fully air-gapped deployments.

## Project Structure

The repository is modularized into discrete services:

- `core_api/`: The main FastAPI service housing the LangGraph state machine, agents, Redis caching logic, and Cypher chains.
- `knowledge_graph_etl/`: Data pipeline scripts to bulk-load and embed entities into the Neo4j instance.
- `web_ui/`: A Streamlit application for end-user chat interactions, consuming the SSE streams.
- `admin_query_portal/`: A secondary Streamlit portal for admins to manually review failing queries, correct them, and upload the validated Text-to-Cypher pairs back into the few-shot vector index to continuously improve the model.

## Setup and Deployment

1. Set up a Neo4j instance (AuraDB or local Docker container).
2. Create a `.env` file in the root directory:

```env
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>

OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

# Example Paths to your raw structured data CSVs
ENTITIES_CSV_PATH=https://raw.githubusercontent.com/<ORG>/<REPO>/main/data/entities.csv
REVIEWS_CSV_PATH=https://raw.githubusercontent.com/<ORG>/<REPO>/main/data/unstructured.csv
EXAMPLE_CYPHER_CSV_PATH=https://raw.githubusercontent.com/<ORG>/<REPO>/main/data/example_cypher.csv

CHATBOT_URL=http://host.docker.internal:8000/graph-rag-agent

ENTERPRISE_AGENT_MODEL=gpt-4o-mini
ENTERPRISE_CYPHER_MODEL=gpt-4o-mini
ENTERPRISE_QA_MODEL=gpt-4o-mini

NEO4J_CYPHER_EXAMPLES_INDEX_NAME=questions
NEO4J_CYPHER_EXAMPLES_NODE_NAME=Question
NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY=question
NEO4J_CYPHER_EXAMPLES_METADATA_NAME=cypher
```

3. Launch the complete microservice stack via Docker Compose:

```console
$ docker-compose up --build
```

- **Core API Docs**: `http://localhost:8000/docs`
- **End-User Chat UI**: `http://localhost:8501/`
- **Admin Query Tuner**: `http://localhost:8502/`

## Evaluation Benchmarks

The system includes an automated evaluation suite (`core_api/eval/eval_cypher.py`) measuring exact match rates across aggregation, filtering, and multi-hop queries.

| Metric | Performance | Notes |
| :--- | :--- | :--- |
| **Exact Match Rate** | 100.0% | Across 36 test benchmarks |
| **Mutation Injection Defense** | 100% Blocked | `CREATE`/`DELETE`/`SET` rejected |
| **Schema Direction Validation** | Pre-execution | Reversed relations rejected |
| **Cached Query Latency** | ~0.10 ms avg | Redis L1 caching |
| **Unbounded Query Guard** | Enforced | `LIMIT 200` ceiling |
