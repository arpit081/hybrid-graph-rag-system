"""
LLM and Embeddings factory supporting both OpenAI and local Ollama providers.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.config import settings

logger = logging.getLogger(__name__)


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Returns a Chat model based on LLM_PROVIDER ('openai' | 'ollama').

    NOTE ON OUTPUT QUALITY DIFFERENCES:
    - Cypher Generation: Smaller local models (e.g. llama3.1, mistral) tend to have lower
      syntax accuracy, may hallucinate relationship types/directions (e.g. [:TREATS] vs [:AT]),
      and occasionally fail complex Cypher aggregations. The LangGraph `validator` node with
      `CypherQueryCorrector` and retry loop is critical when using Ollama. Focus eval testing
      on multi_hop and aggregation queries.
    - Routing & Intent: Local models may be less deterministic in structured tool selection;
      the router's fallback heuristic provides safety.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        # Map default OpenAI model names to Ollama default model if not overridden
        chosen_model = (
            model
            if (model and not model.startswith("gpt-"))
            else settings.OLLAMA_MODEL
        )
        logger.info(
            f"Instantiating ChatOllama with model={chosen_model}, base_url={settings.OLLAMA_BASE_URL}"
        )
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=chosen_model,
            temperature=temperature,
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    chosen_model = model or settings.ENTERPRISE_AGENT_MODEL
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=chosen_model,
        temperature=temperature,
        **kwargs,
    )


def get_embeddings(
    model: Optional[str] = None,
    **kwargs: Any,
) -> Embeddings:
    """Returns an Embeddings instance based on LLM_PROVIDER ('openai' | 'ollama').

    NOTE ON VECTOR EMBEDDING DIFFERENCES:
    - Dimension Mismatch: OpenAI text-embedding-ada-002 outputs 1536 dimensions, whereas
      Ollama's nomic-embed-text outputs 768 dimensions.
    - Database Re-indexing: When switching between OpenAI and Ollama, Neo4j vector indexes
      (e.g. 'reviews', 'questions') MUST be recreated/re-indexed with matching dimensionality,
      otherwise cosine similarity calculations and index queries will error.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        chosen_model = model or settings.OLLAMA_EMBEDDING_MODEL
        logger.info(
            f"Instantiating OllamaEmbeddings with model={chosen_model}, base_url={settings.OLLAMA_BASE_URL}"
        )
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=chosen_model,
            **kwargs,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        **kwargs,
    )
