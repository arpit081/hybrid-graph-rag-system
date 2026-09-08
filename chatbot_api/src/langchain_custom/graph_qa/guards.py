"""
Cypher execution security guards and relationship direction validation.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from langchain_community.chains.graph_qa.cypher_utils import (
        CypherQueryCorrector,
        Schema,
    )
except ImportError:
    class Schema:  # type: ignore[no-redef]
        def __init__(self, start: str, type: str, end: str) -> None:
            self.start = start
            self.type = type
            self.end = end

    class CypherQueryCorrector:  # type: ignore[no-redef]
        def __init__(self, schemas: Any) -> None:
            self.schemas = schemas

        def __call__(self, query: str) -> str:
            return query

try:
    import neo4j
    AccessMode = neo4j.AccessMode
except (ImportError, AttributeError):
    class AccessMode:  # type: ignore[no-redef]
        READ = "READ"
        WRITE = "WRITE"

logger = logging.getLogger(__name__)

FORBIDDEN_CYPHER_KEYWORDS = {
    "CREATE",
    "DELETE",
    "SET",
    "MERGE",
    "DETACH",
    "DROP",
    "REMOVE",
}

FORBIDDEN_KEYWORDS_REGEX = re.compile(
    r"\b(CREATE|DELETE|SET|MERGE|DETACH|DROP|REMOVE)\b", re.IGNORECASE
)

LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)


def strip_string_literals(cypher: str) -> str:
    """Replaces contents of single and double quoted string literals with empty strings."""
    pattern = r"('(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")"
    return re.sub(pattern, "''", cypher)


def apply_query_guard(query: str, default_limit: int = 200) -> str:
    """Validates that a generated Cypher query is read-only and enforces a LIMIT clause.

    - Rejects queries containing mutating keywords (CREATE, DELETE, SET, MERGE, DETACH).
    - Auto-appends a LIMIT clause if none is present.
    """
    if not query or not query.strip():
        raise ValueError("Cypher query cannot be empty.")

    cleaned_code = strip_string_literals(query)

    # 1. Reject mutating keywords outright
    match = FORBIDDEN_KEYWORDS_REGEX.search(cleaned_code)
    if match:
        raise ValueError(
            f"Cypher query rejected by security guard: Mutating keyword '{match.group(0).upper()}' "
            f"is forbidden in read-only mode."
        )

    # 2. Check and enforce LIMIT clause
    query_str = query.strip().rstrip(";").strip()
    if not LIMIT_PATTERN.search(cleaned_code):
        query_str = f"{query_str}\nLIMIT {default_limit}"

    return query_str


class ExtendedCypherQueryCorrector(CypherQueryCorrector):
    """Extends CypherQueryCorrector to validate relationship directions against the schema

    before execution, rejecting reversed relationship paths to trigger regeneration.
    """

    def __init__(self, schemas: List[Schema]) -> None:
        super().__init__(schemas)
        self.schema_triplets: Set[Tuple[str, str, str]] = {
            (s.start.upper(), s.type.upper(), s.end.upper()) for s in schemas
        }

    def validate_relationship_direction(self, query: str) -> Tuple[bool, Optional[str]]:
        """Checks if all directed relationships in the query match schema direction.

        Returns (True, None) if valid, or (False, error_message) if a reversed/invalid
        relationship is detected.
        """
        alias_to_label: Dict[str, str] = {}
        for m in re.finditer(r"\((\w+)\s*:\s*([A-Za-z0-9_]+)", query):
            alias_to_label[m.group(1)] = m.group(2).upper()

        # Check outgoing relationships: (left)-[:REL]->(right)
        outgoing_pattern = re.compile(
            r"\((?P<left_var>\w+)?(?:\s*:\s*(?P<left_lbl>\w+))?[^)]*\)\s*"
            r"-\s*\[(?P<rel_var>\w+)?(?:\s*:\s*(?P<rel_type>\w+))?[^\]]*\]\s*->\s*"
            r"\((?P<right_var>\w+)?(?:\s*:\s*(?P<right_lbl>\w+))?[^)]*\)"
        )
        for m in outgoing_pattern.finditer(query):
            start_lbl = m.group("left_lbl") or alias_to_label.get(m.group("left_var") or "")
            rel_type = m.group("rel_type")
            end_lbl = m.group("right_lbl") or alias_to_label.get(m.group("right_var") or "")

            if start_lbl and rel_type and end_lbl:
                trip = (start_lbl.upper(), rel_type.upper(), end_lbl.upper())
                rev = (end_lbl.upper(), rel_type.upper(), start_lbl.upper())
                if trip not in self.schema_triplets and rev in self.schema_triplets:
                    err = (
                        f"Reversed relationship direction detected: ({start_lbl})-[:{rel_type}]->({end_lbl}) "
                        f"does not exist in schema. Valid direction is ({end_lbl})-[:{rel_type}]->({start_lbl})."
                    )
                    return False, err

        # Check incoming relationships: (left)<-[:REL]-(right)
        incoming_pattern = re.compile(
            r"\((?P<left_var>\w+)?(?:\s*:\s*(?P<left_lbl>\w+))?[^)]*\)\s*"
            r"<-\s*\[(?P<rel_var>\w+)?(?:\s*:\s*(?P<rel_type>\w+))?[^\]]*\]\s*-\s*"
            r"\((?P<right_var>\w+)?(?:\s*:\s*(?P<right_lbl>\w+))?[^)]*\)"
        )
        for m in incoming_pattern.finditer(query):
            start_lbl = m.group("right_lbl") or alias_to_label.get(m.group("right_var") or "")
            rel_type = m.group("rel_type")
            end_lbl = m.group("left_lbl") or alias_to_label.get(m.group("left_var") or "")

            if start_lbl and rel_type and end_lbl:
                trip = (start_lbl.upper(), rel_type.upper(), end_lbl.upper())
                rev = (end_lbl.upper(), rel_type.upper(), start_lbl.upper())
                if trip not in self.schema_triplets and rev in self.schema_triplets:
                    err = (
                        f"Reversed relationship direction detected: ({end_lbl})<-[:{rel_type}]-({start_lbl}) "
                        f"does not exist in schema. Valid direction is ({start_lbl})-[:{rel_type}]->({end_lbl})."
                    )
                    return False, err

        return True, None

    def __call__(self, query: str) -> str:
        """Validates relationship directions, raising ValueError on reversed relationships to trigger regeneration."""
        is_valid, err = self.validate_relationship_direction(query)
        if not is_valid:
            raise ValueError(err)

        return super().__call__(query)


def execute_read_only_cypher(
    graph: Any, query: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Forces execution of generated Cypher within a Neo4j session strictly in read-only access mode."""
    driver = getattr(graph, "_driver", None)
    database = getattr(graph, "_database", "neo4j")

    if driver is not None:
        with driver.session(
            database=database,
            default_access_mode=AccessMode.READ,
        ) as session:
            def _read_tx(tx: Any) -> List[Dict[str, Any]]:
                res = tx.run(query, params or {})
                return [record.data() for record in res]

            return session.execute_read(_read_tx)

    # Fallback to graph.query if driver is absent/mocked
    return graph.query(query, params or {})
