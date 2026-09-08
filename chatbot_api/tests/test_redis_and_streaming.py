import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure chatbot_api is on sys.path for direct invocations
CHATBOT_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CHATBOT_API_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_API_DIR)

# Dummy env vars so config and module imports succeed
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "secret")
os.environ.setdefault("OPENAI_API_KEY", "sk-test123")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

import types
from importlib.machinery import ModuleSpec

class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattribute__(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock

class MockFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in ["langchain", "langchain_core", "langchain_community", "langchain_openai", "langgraph", "neo4j"]):
            return ModuleSpec(fullname, cls, is_package=True)
        return None

    @classmethod
    def create_module(cls, spec):
        return MockModule(spec.name)

    @classmethod
    def exec_module(cls, module):
        if module.__name__ == "langchain.chains.base":
            module.Chain = type(
                "Chain",
                (),
                {
                    "__init__": lambda self, *a, **k: None,
                    "invoke": MagicMock(),
                    "run": MagicMock(),
                },
            )
        elif module.__name__ == "langgraph.graph":
            module.END = "__end__"

sys.meta_path.insert(0, MockFinder)

from fastapi.testclient import TestClient
from src.main import app
from src.models.hospital_rag_query import HospitalQueryInput
from src.utils.redis_cache import (
    get_cache_key,
    get_cached_cypher,
    normalize_question,
    reset_redis_client,
    set_cached_cypher,
    set_test_redis_client,
)


class MockRedis:
    """In-memory Redis mock for unit testing."""

    def __init__(self):
        self.store = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex=None):
        self.store[key] = value

    def ping(self):
        return True


class TestRedisCacheAndStreaming(unittest.TestCase):
    def setUp(self):
        self.mock_redis = MockRedis()
        set_test_redis_client(self.mock_redis)

    def tearDown(self):
        reset_redis_client()

    def test_normalize_question(self):
        """Question normalization must be invariant to case, punctuation, whitespace, and quotes."""
        q1 = "Which hospitals are in the hospital system?"
        q2 = "  which  hospitals are in the hospital system "
        q3 = '"Which hospitals are in the hospital system???"'
        q4 = "'which hospitals are in the hospital system!'"

        expected = "which hospitals are in the hospital system"
        self.assertEqual(normalize_question(q1), expected)
        self.assertEqual(normalize_question(q2), expected)
        self.assertEqual(normalize_question(q3), expected)
        self.assertEqual(normalize_question(q4), expected)

        key1 = get_cache_key(q1)
        key2 = get_cache_key(q2)
        self.assertEqual(key1, key2)
        self.assertEqual(key1, f"cypher_cache:{expected}")

    def test_redis_cache_get_and_set(self):
        """Storing a query under a question should allow retrieval with formatted variants."""
        question = "What is the average billing amount for Medicaid visits?"
        cypher_query = "MATCH (v:Visit)-[:COVERED_BY]->(p:Payer) WHERE p.name = 'Medicaid' RETURN avg(v.billing_amount) LIMIT 200"

        # Initially absent
        self.assertIsNone(get_cached_cypher(question))

        # Store in cache
        set_cached_cypher(question, cypher_query)

        # Retrieve with identical question
        cached = get_cached_cypher(question)
        self.assertEqual(cached, cypher_query)

        # Retrieve with variant punctuation and whitespace
        variant_question = "  what is the average billing amount for medicaid visits?  "
        cached_variant = get_cached_cypher(variant_question)
        self.assertEqual(cached_variant, cypher_query)

    def test_cypher_gen_checks_cache_before_llm(self):
        """Agent's cypher_gen node must return cached Cypher without invoking LLM."""
        from src.agents.hospital_rag_agent import cypher_gen
        from src.chains.hospital_cypher_chain import hospital_cypher_chain

        question = "Which hospitals are in the hospital system?"
        cached_cypher = "MATCH (h:Hospital) RETURN h.name LIMIT 200"
        set_cached_cypher(question, cached_cypher)

        mock_chain = MagicMock()
        hospital_cypher_chain.cypher_generation_chain = mock_chain

        state = {
            "input": question,
            "output": "",
            "intermediate_steps": [],
            "route": "explore_hospital_database",
            "extracted_hospital": None,
            "cypher_query": None,
            "cypher_context": None,
            "cypher_error": None,
            "cypher_retry_count": 0,
            "validation_passed": False,
            "tool_output": None,
        }
        res = cypher_gen(state)

        # LLM chain must NOT be called on cache hit
        mock_chain.invoke.assert_not_called()
        self.assertEqual(res["cypher_query"], cached_cypher)
        self.assertIsNone(res["cypher_error"])

    def test_non_streaming_fallback_path(self):
        """POST /hospital-rag-agent with stream=False must return standard JSON response."""
        client = TestClient(app)
        mock_response = {
            "input": "How many hospitals are there?",
            "output": "There are 5 hospitals in the system.",
            "intermediate_steps": ["Tool: explore_hospital_database"],
        }

        with patch("src.main.invoke_agent_with_retry", new=AsyncMock(return_value=mock_response)):
            response = client.post(
                "/hospital-rag-agent",
                json={"text": "How many hospitals are there?", "stream": False},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("application/json", response.headers["content-type"])
            data = response.json()
            self.assertEqual(data["output"], "There are 5 hospitals in the system.")
            self.assertEqual(data["input"], "How many hospitals are there?")
            self.assertIn("Tool: explore_hospital_database", data["intermediate_steps"])

    def test_streaming_endpoint_sse(self):
        """POST /hospital-rag-agent with stream=True must return SSE stream with tokens."""
        client = TestClient(app)

        async def mock_event_generator(query: str):
            yield {"type": "status", "stage": "route_selected", "route": "explore_hospital_database"}
            yield {"type": "token", "token": "There "}
            yield {"type": "token", "token": "are "}
            yield {"type": "token", "token": "5 hospitals."}
            yield {
                "type": "done",
                "output": "There are 5 hospitals.",
                "intermediate_steps": ["Tool: explore_hospital_database"],
            }

        with patch("src.main.astream_hospital_rag_agent", side_effect=mock_event_generator):
            response = client.post(
                "/hospital-rag-agent",
                json={"text": "List hospitals", "stream": True},
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])

            body_lines = response.text.strip().split("\n\n")
            tokens = []
            done_payload = None

            for line in body_lines:
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    if payload.get("type") == "token":
                        tokens.append(payload["token"])
                    elif payload.get("type") == "done":
                        done_payload = payload

            self.assertEqual("".join(tokens), "There are 5 hospitals.")
            self.assertIsNotNone(done_payload)
            self.assertEqual(done_payload["output"], "There are 5 hospitals.")
            self.assertIn("Tool: explore_hospital_database", done_payload["intermediate_steps"])


if __name__ == "__main__":
    unittest.main()
