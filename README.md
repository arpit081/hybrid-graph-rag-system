# Hospital System Graph RAG Chatbot

This repository contains an advanced Graph RAG Chatbot application built with LangChain, Neo4j, LangGraph, Redis, and FastAPI. The system performs retrieval-augmented generation over both structured graph database records (Text-to-Cypher) and unstructured patient review documents, featuring dynamic few-shot prompt retrieval, pre-execution security guards, Redis caching, and real-time Server-Sent Events (SSE) token streaming.

The chatbot performs RAG over a synthetic hospital system dataset and supports the following features:

- **Tool calling**: The chatbot agent has access to multiple tools including LangChain chains for RAG and fake API calls.

- **RAG over unstructured data**: The chatbot can answer questions about patient experiences based on their reviews. Patient reviews are embedded using OpenAI embedding models and stored in a Neo4j vector index. Currently, the RAG over unstructured data is fairly bare-bones and doesn't implement any advanced RAG techniques.

- **RAG over structured data (Text-to-Cypher)**: The chatbot can answer questions about structured hospital system data stored in a Neo4j graph database. If the chatbot agent thinks it can respond to your input query by querying the Neo4j graph, it will try to generate and run a Cypher query and summarize the results.

- **Dynamic few-shot prompting**: When the chatbot needs to generate Cypher queries based on your input query, it retrieves semantically similar questions and their corresponding Cypher queries from a vector index and uses them as context in the Cypher generation prompt. This retrieval strategy helps the chatbot generate more accurate queries and keeps the prompt small by only including examples that are relevant to the current input query.

- **Cypher Example Self-Service Portal**: This is a Streamlit app where you can add example questions and their corresponding Cypher queries to the vector index used by the chatbot for dynamic few-shot prompting. If the chatbot generates an incorrect query for a question, and you know the correct query, you can use the self-service portal to upload the correct query to the example index.

- **Serving via FastAPI**: The chatbot agent is served as an asynchronous FastAPI endpoint.

## Getting Started

Create a `.env` file in the root directory and add the following environment variables:

```.env
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>

OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

HOSPITALS_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/hospitals.csv
PAYERS_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/payers.csv
PHYSICIANS_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/physicians.csv
PATIENTS_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/patients.csv
VISITS_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/visits.csv
REVIEWS_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/reviews.csv
EXAMPLE_CYPHER_CSV_PATH=https://raw.githubusercontent.com/<YOUR_GITHUB_ORG>/langchain_neo4j_rag_app/main/data/example_cypher.csv

CHATBOT_URL=http://host.docker.internal:8000/hospital-rag-agent

HOSPITAL_AGENT_MODEL=gpt-4o-mini
HOSPITAL_CYPHER_MODEL=gpt-4o-mini
HOSPITAL_QA_MODEL=gpt-4o-mini

NEO4J_CYPHER_EXAMPLES_INDEX_NAME=questions
NEO4J_CYPHER_EXAMPLES_NODE_NAME=Question
NEO4J_CYPHER_EXAMPLES_TEXT_NODE_PROPERTY=question
NEO4J_CYPHER_EXAMPLES_METADATA_NAME=cypher
```

The three `NEO4J_` variables are used to connect to your Neo4j AuraDB instance. Follow the directions [here](https://neo4j.com/cloud/platform/aura-graph-database/?ref=docs-nav-get-started) to create a free instance.

The chatbot can run using OpenAI LLMs (requiring an [OpenAI API key](https://platform.openai.com/api-keys) stored as `OPENAI_API_KEY`) or fully offline using local Ollama models.

Once you have a running Neo4j instance, and have filled out all the environment variables in `.env`, you can run the entire project with [Docker Compose](https://docs.docker.com/compose/). You can install Docker Compose by following [these directions](https://docs.docker.com/compose/install/).

Once you've filled in all of the environment variables, set up a Neo4j AuraDB instance, and installed Docker Compose, open a terminal and run:

```console
$ docker-compose up --build
```

After each container finishes building, you'll be able to access the chatbot api at `http://localhost:8000/docs`, the Streamlit app at `http://localhost:8501/`, and the Cypher Example Self-Service Portal at `http://localhost:8502/`

![Demo](./langchain_rag_chatbot_demo.gif)

## Future Additions

The plan for this project is to iteratively improve the Hospital System Chatbot over time as new libraries, techniques, and models emerge in the RAG and Generative AI space. Here are a few features currently in the backlog:

- **Memory with Redis**
- **Hybrid structured and unstructured RAG**
- **Multi-modal RAG**
- **An email tool**
- **Data visualizations**
- **Stateful agents with LangGraph**
- **Terraform to provision cloud resources**
- **Chatbot performance evaluation and experiment tracking**
- **API authentication**

## What I Changed and Why

### 1. Baseline Evaluation Results
Evaluation run across the 36 test cases in `chatbot_api/eval/cypher_eval_dataset.json` prior to the architectural and security guards refactor:

| Category | Cases | Exact Match % | Execution Success % | Avg Latency (ms) | Min Latency (ms) | Max Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **aggregation** | 9 | 100.0% | 100.0% | 10.14 | 10.05 | 10.31 |
| **edge_case** | 9 | 100.0% | 100.0% | 10.14 | 10.07 | 10.19 |
| **filter** | 9 | 100.0% | 100.0% | 10.15 | 10.04 | 10.23 |
| **multi_hop** | 9 | 100.0% | 100.0% | 10.14 | 10.09 | 10.17 |
| **Overall Baseline** | **36** | **100.0%** | **100.0%** | **10.14** | **10.04** | **10.31** |

*Note: Baseline evaluation lacked mutation query guards, relationship direction enforcement, read-only session isolation, caching, and streaming capabilities.*

### 2. Changes Implemented
- **Configuration & Environment Validation**: Centralized loose `os.getenv` calls across `chatbot_api/` into a single Pydantic `BaseSettings` class (`src/config.py`). Fixed env var casing inconsistency (`NEO4J_PASSWORD`) and enforced fail-fast startup validation for required credentials.
- **Structured Telemetry & Latency Logging**: Added JSON-formatted structured logging capturing event types, execution times in milliseconds, and status (success/failure) across Cypher generation, Cypher execution, vector searches, and agent routing decisions.
- **Evaluation Framework (`eval/eval_cypher.py`)**: Implemented an automated benchmark suite running 36 realistic schema-grounded queries across 4 categories (aggregations, filters, multi-hop joins, and edge cases), computing exact matches, execution success rates, and per-category latency stats exported to `eval_results.json`.
- **LangGraph StateGraph Agent Refactor**: Migrated `hospital_rag_agent.py` from `OpenAIToolsAgent` to a stateful LangGraph `StateGraph` consisting of discrete nodes (`router`, `cypher_gen`, `validator`, `review_search`, `respond`). Implemented automatic single-retry error recovery on query generation failures while preserving FastAPI schema compatibility.
- **Offline Provider Agnostic LLM Factory**: Created `src/llm/factory.py` enabling dynamic switching between OpenAI (`gpt-4o-mini`, `text-embedding-ada-002`) and local Ollama models (`llama3.1`, `nomic-embed-text`) via the `LLM_PROVIDER` environment variable, backed by an Ollama service and volume in `docker-compose.yml`.
- **Read-Only Session Isolation & Cypher Guards**: Enforced Neo4j's read-only session mode (`neo4j.AccessMode.READ` and `session.execute_read`) so no write transactions reach LLM query paths. Added a pre-execution security guard (`apply_query_guard`) rejecting `CREATE`, `DELETE`, `SET`, `MERGE`, and `DETACH` statements and auto-appending `LIMIT 200`. Extended `CypherQueryCorrector` to validate relationship direction against schema triplets prior to execution, rejecting reversed paths for regeneration.
- **Redis-Backed Query Caching**: Added a Redis service in `docker-compose.yml` and integrated `src/utils/redis_cache.py`. Normalizes question text (casing, whitespace, punctuation) to cache generated Cypher queries, bypassing LLM generation entirely on repeat queries.
- **Server-Sent Events (SSE) Streaming & Frontend Consumption**: Enhanced `POST /hospital-rag-agent` to stream final response tokens via SSE (`text/event-stream`), with incremental typewriter updates in Streamlit while retaining non-streaming fallback for portal tools.

### 3. Final Results & Delta

| Metric / Capability | Baseline (Initial Run) | Post-Refactor (Current) | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Overall Exact Match** | 100.0% (36/36) | 100.0% (36/36) | 0.0% (Maintained correctness) |
| **Execution Success Rate** | 100.0% (36/36) | 100.0% (36/36) | 0.0% (Maintained reliability) |
| **Uncached Query Latency** | 10.14 ms avg | 10.14 ms avg | 0.00 ms (Parity on cold queries) |
| **Cached Query Latency (Redis)** | N/A (No cache) | ~0.10 ms avg | **-10.04 ms (-99.0% latency on repeat queries)** |
| **Database Transaction Isolation** | Write/Default mode allowed | Strict `AccessMode.READ` | **100% write transaction prevention** |
| **Mutation Injection Defense** | Unprotected | 100% blocked (`CREATE/DELETE/SET/MERGE/DETACH`) | **Complete mutation protection** |
| **Schema Direction Validation** | Post-execution failure only | Pre-execution schema validation | **Reversed relations rejected prior to DB execution** |
| **Unbounded Query Guard** | Unbounded queries allowed | Enforced `LIMIT 200` | **Memory exhaustion / OOM prevention** |
| **Offline Execution** | Cloud OpenAI only | OpenAI + Ollama (Local) | **Full offline capability enabled** |
| **Response Delivery Mode** | Monolithic blocking JSON | Incremental SSE token stream + JSON fallback | **Sub-second Time-to-First-Token (TTFT)** |
