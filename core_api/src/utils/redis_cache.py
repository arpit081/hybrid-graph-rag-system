"""
Redis-backed cache for generated Cypher queries.
Keys on normalized user question text and stores generated Cypher queries.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)

_redis_client: Optional[Any] = None
_client_initialized: bool = False


def normalize_question(question: str) -> str:
    """
    Normalizes user question for consistent cache keying:
    - Lowercase
    - Strips leading/trailing whitespace, punctuation, and quotes
    - Collapses multiple whitespace characters to a single space
    """
    if not question:
        return ""
    q = question.strip().lower()
    # Strip surrounding quotation marks
    q = re.sub(r"^['\"]+|['\"]+$", "", q)
    # Strip trailing punctuation (. ? ! ;)
    q = re.sub(r"[\.?!;]+$", "", q).strip()
    # Collapse internal whitespace
    q = re.sub(r"\s+", " ", q)
    return q


def get_cache_key(question: str) -> str:
    """Constructs Redis key from normalized question."""
    normalized = normalize_question(question)
    return f"cypher_cache:{normalized}"


def get_redis_client() -> Optional[Any]:
    """Returns a connected Redis client, or None if unavailable/unconfigured."""
    global _redis_client, _client_initialized

    if _client_initialized:
        return _redis_client

    try:
        import redis

        client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        client.ping()
        _redis_client = client
        _client_initialized = True
        return _redis_client
    except Exception as e:
        logger.debug(f"Redis is not available ({e}); proceeding without Cypher cache.")
        _redis_client = None
        _client_initialized = True
        return None


def set_test_redis_client(client: Optional[Any]) -> None:
    """Allows test fixtures to inject a mock Redis client."""
    global _redis_client, _client_initialized
    _redis_client = client
    _client_initialized = True


def reset_redis_client() -> None:
    """Resets client state to allow re-initialization."""
    global _redis_client, _client_initialized
    _redis_client = None
    _client_initialized = False


def get_cached_cypher(question: str) -> Optional[str]:
    """Retrieves cached Cypher query for the question, if present."""
    client = get_redis_client()
    if client is None:
        return None

    key = get_cache_key(question)
    try:
        cached_query = client.get(key)
        if cached_query:
            logger.info(
                "Redis Cypher cache hit",
                extra={
                    "event": "cypher_cache_hit",
                    "cache_key": key,
                    "question": question,
                    "cypher_query": cached_query,
                },
            )
            return cached_query
    except Exception as e:
        logger.warning(f"Error reading from Redis Cypher cache: {e}")

    return None


def set_cached_cypher(
    question: str, cypher_query: str, ttl: Optional[int] = None
) -> None:
    """Stores generated Cypher query in Redis cache under normalized question key."""
    if not question or not cypher_query:
        return

    client = get_redis_client()
    if client is None:
        return

    key = get_cache_key(question)
    expiry = ttl or getattr(settings, "REDIS_CYPHER_CACHE_TTL", 86400)
    try:
        client.set(key, cypher_query, ex=expiry)
        logger.info(
            "Cached Cypher query in Redis",
            extra={
                "event": "cypher_cache_set",
                "cache_key": key,
                "question": question,
                "ttl": expiry,
            },
        )
    except Exception as e:
        logger.warning(f"Error writing to Redis Cypher cache: {e}")
