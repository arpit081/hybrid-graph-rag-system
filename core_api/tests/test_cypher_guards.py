import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure core_api is on sys.path for direct invocations
CHATBOT_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CHATBOT_API_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_API_DIR)

try:
    import pytest
except ImportError:
    pytest = None

from src.langchain_custom.graph_qa.guards import (
    AccessMode,
    ExtendedCypherQueryCorrector,
    Schema,
    apply_query_guard,
    execute_read_only_cypher,
)


class TestCypherGuards(unittest.TestCase):
    def setUp(self):
        self.schema = [
            Schema("Visit", "AT", "Hospital"),
            Schema("Visit", "WRITES", "Review"),
            Schema("Physician", "TREATS", "Visit"),
            Schema("Visit", "COVERED_BY", "Payer"),
            Schema("Patient", "HAS", "Visit"),
            Schema("Hospital", "EMPLOYS", "Physician"),
        ]
        self.corrector = ExtendedCypherQueryCorrector(self.schema)

    def test_query_guard_rejects_mutations(self):
        """Query guard must reject CREATE, DELETE, SET, MERGE, DETACH keywords outright."""
        malicious_queries = [
            "MATCH (h:Hospital) DELETE h",
            "MATCH (h:Hospital) DETACH DELETE h",
            "CREATE (h:Hospital {name: 'Fake Hospital'})",
            "MATCH (h:Hospital) SET h.name = 'Compromised'",
            "MERGE (p:Patient {id: 9999})",
            "MATCH (v:Visit) DROP CONSTRAINT unique_id",
            "MATCH (n) REMOVE n:Hospital",
        ]
        for query in malicious_queries:
            with self.assertRaises(ValueError, msg=f"Should have rejected: {query}"):
                apply_query_guard(query)

    def test_query_guard_allows_safe_strings_with_mutation_words(self):
        """Query guard must not falsely reject queries where keywords appear only inside string literals."""
        safe_queries = [
            "MATCH (p:Patient) WHERE p.name = 'SET' RETURN p.id",
            "MATCH (r:Review) WHERE r.text CONTAINS 'please delete my review' RETURN r.id",
            "MATCH (v:Visit) WHERE v.chief_complaint = 'create swelling' RETURN v.id",
            "MATCH (p:Patient) WHERE p.status = 'asset' RETURN p.name",
        ]
        for query in safe_queries:
            guarded = apply_query_guard(query)
            self.assertIn("LIMIT", guarded)

    def test_query_guard_enforces_limit(self):
        """Query guard must auto-append LIMIT 200 when absent, and preserve existing LIMIT."""
        query_without_limit = "MATCH (h:Hospital) RETURN h.name"
        guarded = apply_query_guard(query_without_limit, default_limit=200)
        self.assertTrue(guarded.endswith("LIMIT 200"))

        query_with_limit = "MATCH (h:Hospital) RETURN h.name LIMIT 50"
        guarded_preserved = apply_query_guard(query_with_limit, default_limit=200)
        self.assertTrue(guarded_preserved.endswith("LIMIT 50"))

    def test_extended_corrector_validates_relationship_direction(self):
        """ExtendedCypherQueryCorrector must reject queries with reversed relationship directions."""
        # Hospital -> AT -> Visit is reversed (schema is Visit -> AT -> Hospital)
        reversed_at = "MATCH (h:Hospital)-[:AT]->(v:Visit) RETURN h"
        is_valid, err = self.corrector.validate_relationship_direction(reversed_at)
        self.assertFalse(is_valid)
        self.assertIn("Reversed relationship direction", err)
        with self.assertRaises(ValueError):
            self.corrector(reversed_at)

        # Visit -> AT -> Hospital is valid
        valid_at = "MATCH (v:Visit)-[:AT]->(h:Hospital) RETURN v"
        is_valid, err = self.corrector.validate_relationship_direction(valid_at)
        self.assertTrue(is_valid)
        self.assertIsNone(err)

        # Visit -> TREATS -> Physician is reversed (schema is Physician -> TREATS -> Visit)
        reversed_treats = "MATCH (v:Visit)-[:TREATS]->(p:Physician) RETURN v"
        is_valid, err = self.corrector.validate_relationship_direction(reversed_treats)
        self.assertFalse(is_valid)
        with self.assertRaises(ValueError):
            self.corrector(reversed_treats)

        # Incoming arrow check: (v:Visit)<-[:AT]-(h:Hospital) is reversed
        reversed_incoming = "MATCH (v:Visit)<-[:AT]-(h:Hospital) RETURN v"
        is_valid, err = self.corrector.validate_relationship_direction(reversed_incoming)
        self.assertFalse(is_valid)

    def test_read_only_access_mode_enforcement(self):
        """execute_read_only_cypher must open session with AccessMode.READ and use execute_read."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.execute_read.return_value = [{"count": 42}]

        class MockGraph:
            _driver = mock_driver
            _database = "neo4j"

        result = execute_read_only_cypher(MockGraph(), "MATCH (n) RETURN count(n)")

        mock_driver.session.assert_called_once_with(
            database="neo4j",
            default_access_mode=AccessMode.READ,
        )
        mock_session.execute_read.assert_called_once()
        self.assertEqual(result, [{"count": 42}])


if __name__ == "__main__":
    unittest.main()
