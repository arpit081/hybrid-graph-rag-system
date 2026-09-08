import io
import json
import logging
import os
import unittest

# Provide dummy env vars so the initial module import of src.config succeeds
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "secret")
os.environ.setdefault("OPENAI_API_KEY", "sk-test123")

from pydantic import ValidationError
from src.config import Settings
from src.utils.logging_config import JSONFormatter, wrap_vector_store_with_logging


class TestConfigAndLogging(unittest.TestCase):
    def test_settings_fail_fast_on_missing_required_fields(self):
        """Settings must fail fast at startup if required fields are missing."""
        env_backup = {}
        keys = [
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "OPENAI_API_KEY",
            "HOSPITAL_AGENT_MODEL",
            "HOSPITAL_CYPHER_MODEL",
            "HOSPITAL_QA_MODEL",
        ]
        for key in keys:
            if key in os.environ:
                env_backup[key] = os.environ.pop(key)

        try:
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=None)

            errors = ctx.exception.errors()
            missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
            self.assertIn("NEO4J_URI", missing_fields)
            self.assertIn("NEO4J_USERNAME", missing_fields)
            self.assertIn("NEO4J_PASSWORD", missing_fields)
            self.assertIn("OPENAI_API_KEY", missing_fields)
        finally:
            os.environ.update(env_backup)

    def test_settings_valid_and_defaults(self):
        """Settings should validate and apply defaults when required fields are present."""
        env_backup = {}
        test_env = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "secret",
            "OPENAI_API_KEY": "sk-test123",
        }
        for key, val in test_env.items():
            if key in os.environ:
                env_backup[key] = os.environ[key]
            os.environ[key] = val

        try:
            s = Settings(_env_file=None)
            self.assertEqual(s.NEO4J_URI, "bolt://localhost:7687")
            self.assertEqual(s.NEO4J_USERNAME, "neo4j")
            self.assertEqual(s.NEO4J_PASSWORD, "secret")
            self.assertEqual(s.OPENAI_API_KEY, "sk-test123")
            self.assertEqual(s.HOSPITAL_AGENT_MODEL, "gpt-4o-mini")
            self.assertEqual(s.NEO4J_CYPHER_EXAMPLES_INDEX_NAME, "questions")
        finally:
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)

    def test_settings_ollama_provider_no_openai_key_needed(self):
        """When LLM_PROVIDER is ollama, OPENAI_API_KEY is not required."""
        env_backup = {}
        test_env = {
            "LLM_PROVIDER": "ollama",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "secret",
        }
        if "OPENAI_API_KEY" in os.environ:
            env_backup["OPENAI_API_KEY"] = os.environ.pop("OPENAI_API_KEY")

        for key, val in test_env.items():
            if key in os.environ:
                env_backup[key] = os.environ[key]
            os.environ[key] = val

        try:
            s = Settings(_env_file=None)
            self.assertEqual(s.LLM_PROVIDER, "ollama")
            self.assertIsNone(s.OPENAI_API_KEY)
            self.assertEqual(s.OLLAMA_MODEL, "llama3.1")
            self.assertEqual(s.OLLAMA_EMBEDDING_MODEL, "nomic-embed-text")
        finally:
            for key in test_env:
                if key in env_backup:
                    os.environ[key] = env_backup[key]
                else:
                    os.environ.pop(key, None)
            if "OPENAI_API_KEY" in env_backup:
                os.environ["OPENAI_API_KEY"] = env_backup["OPENAI_API_KEY"]

    def test_json_formatter_structured_logging(self):
        """JSONFormatter must output valid JSON containing message and custom structured extras."""
        formatter = JSONFormatter()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        test_logger = logging.getLogger("test_structured_logger")
        test_logger.setLevel(logging.INFO)
        test_logger.handlers = [handler]
        test_logger.propagate = False

        test_logger.info(
            "Cypher execution completed",
            extra={
                "event": "cypher_execution",
                "cypher_query": "MATCH (h:Hospital) RETURN h.name",
                "execution_time_ms": 12.34,
                "status": "success",
            },
        )

        output = stream.getvalue().strip()
        data = json.loads(output)

        self.assertEqual(data["message"], "Cypher execution completed")
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["event"], "cypher_execution")
        self.assertEqual(data["cypher_query"], "MATCH (h:Hospital) RETURN h.name")
        self.assertEqual(data["execution_time_ms"], 12.34)
        self.assertEqual(data["status"], "success")
        self.assertIn("timestamp", data)

    def test_wrap_vector_store_with_logging(self):
        """wrap_vector_store_with_logging must log search queries and execution time in ms."""
        formatter = JSONFormatter()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        vec_logger = logging.getLogger("vector_search")
        vec_logger.setLevel(logging.INFO)
        vec_logger.handlers = [handler]
        vec_logger.propagate = False

        class DummyVectorStore:
            def similarity_search(self, query, **kwargs):
                return [{"mock": "result"}]

        store = DummyVectorStore()
        wrapped_store = wrap_vector_store_with_logging(store, "test_index")
        results = wrapped_store.similarity_search("find hospital")

        self.assertEqual(len(results), 1)
        output = stream.getvalue().strip()
        data = json.loads(output)

        self.assertEqual(data["event"], "vector_search")
        self.assertEqual(data["index_name"], "test_index")
        self.assertEqual(data["query"], "find hospital")
        self.assertEqual(data["status"], "success")
        self.assertIn("execution_time_ms", data)
        self.assertEqual(data["result_count"], 1)


if __name__ == "__main__":
    unittest.main()
