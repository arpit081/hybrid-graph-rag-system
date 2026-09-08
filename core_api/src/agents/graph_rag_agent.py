"""
Hospital RAG Agent implemented as a LangGraph StateGraph.

```mermaid
flowchart TD
    START([Start]) --> router[router]
    router -->|explore_hospital_database| cypher_gen[cypher_gen]
    router -->|explore_patient_experiences| review_search[review_search]
    router -->|get_hospital_wait_time| get_hospital_wait_time[get_hospital_wait_time]
    router -->|find_most_available_hospital| find_most_available_hospital[find_most_available_hospital]

    cypher_gen --> validator[validator]
    validator -->|valid / success| respond[respond]
    validator -->|failure & retry < 1| cypher_gen
    validator -->|failure & retry >= 1| respond

    review_search --> respond
    get_hospital_wait_time --> respond
    find_most_available_hospital --> respond

    respond --> END([End])
```
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.config import settings
from src.chains.enterprise_cypher_chain import enterprise_cypher_chain
from src.chains.hospital_review_chain import review_prompt, reviews_vector_chain
from src.llm import get_llm
from src.langchain_custom.graph_qa.cypher import extract_cypher, remove_keys_from_dicts
from src.langchain_custom.graph_qa.guards import (
    apply_query_guard,
    execute_read_only_cypher,
)
from src.tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)
from src.utils.redis_cache import get_cached_cypher, set_cached_cypher

logger = logging.getLogger(__name__)

agent_chat_model = get_llm(
    model=settings.ENTERPRISE_AGENT_MODEL,
    temperature=0,
)


class AgentActionStep:
    """Action representation maintaining backward compatibility with LangChain agent steps."""

    def __init__(self, tool: str, tool_input: Any, log: str = "") -> None:
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

    def __repr__(self) -> str:
        return f"AgentAction(tool='{self.tool}', tool_input={repr(self.tool_input)})"

    def __str__(self) -> str:
        return f"Tool: {self.tool}, Input: {self.tool_input}"


class AgentState(TypedDict):
    input: str
    output: str
    intermediate_steps: List[Any]
    route: str
    extracted_hospital: Optional[str]
    cypher_query: Optional[str]
    cypher_context: Optional[Any]
    cypher_error: Optional[str]
    cypher_retry_count: int
    validation_passed: bool
    tool_output: Optional[Any]


class RouteDecision(BaseModel):
    destination: Literal[
        "explore_hospital_database",
        "explore_patient_experiences",
        "get_hospital_wait_time",
        "find_most_available_hospital",
    ] = Field(
        description="The appropriate path for handling the user query."
    )
    hospital_name: Optional[str] = Field(
        default=None,
        description="Extracted hospital name when asking for a specific hospital wait time.",
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a routing agent for a hospital healthcare system.
Analyze the user's input and select the most appropriate destination path:

1. 'explore_hospital_database': Questions about database entities, structured queries, counts, averages, totals, comparisons, statistics, physicians, patients, visits, admissions, diagnoses, billing amounts, insurance payers, or hospitals.
2. 'explore_patient_experiences': Questions asking about patient experiences, feelings, satisfaction, reviews, complaints, or qualitative doctor/hospital feedback.
3. 'get_hospital_wait_time': Questions asking for the current wait time at a specific named hospital.
4. 'find_most_available_hospital': Questions asking which hospital has the shortest wait time, minimum wait time, or highest current availability.
""",
        ),
        ("user", "{input}"),
    ]
)


def router(state: AgentState) -> Dict[str, Any]:
    """Decides which tool/path to use."""
    user_input = state["input"]
    start_time = time.perf_counter()

    try:
        router_chain = ROUTER_PROMPT | agent_chat_model.with_structured_output(RouteDecision)
        decision = router_chain.invoke({"input": user_input})
        route = decision.destination
        hospital = decision.hospital_name
    except Exception as e:
        logger.warning(f"Structured router failed, using heuristic classification: {e}")
        lower = user_input.lower()
        hospital = None
        if "shortest wait" in lower or "most available" in lower:
            route = "find_most_available_hospital"
        elif "wait time" in lower:
            route = "get_hospital_wait_time"
        elif any(
            w in lower
            for w in [
                "patient said",
                "patients said",
                "patient say",
                "patients say",
                "experience",
                "feeling",
                "satisf",
                "review",
            ]
        ):
            route = "explore_patient_experiences"
        else:
            route = "explore_hospital_database"

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "Router completed path selection",
        extra={
            "event": "agent_tool_selection",
            "tool_name": route,
            "tool_input": user_input,
            "execution_time_ms": duration_ms,
        },
    )

    return {
        "route": route,
        "extracted_hospital": hospital,
        "cypher_retry_count": 0,
        "cypher_error": None,
        "validation_passed": False,
        "intermediate_steps": [],
    }


def route_decision(state: AgentState) -> str:
    """Conditional edge router function."""
    return state.get("route", "explore_hospital_database")


def cypher_gen(state: AgentState) -> Dict[str, Any]:
    """Generates a Cypher query using enterprise_cypher_chain's generation chain."""
    question = state["input"]
    error_context = state.get("cypher_error")
    start_time = time.perf_counter()

    # Check Redis cache before calling LLM (only on initial attempt, not on retry)
    if not error_context:
        cached = get_cached_cypher(question)
        if cached:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "Cypher query retrieved from Redis cache in cypher_gen",
                extra={
                    "event": "cypher_cache_hit",
                    "status": "cache_hit",
                    "execution_time_ms": duration_ms,
                    "cypher_query": cached,
                    "question": question,
                },
            )
            return {
                "cypher_query": cached,
                "cypher_error": None,
            }

    if error_context:
        prompt_question = (
            f"{question}\nNote: Previous Cypher generation/validation failed with: "
            f"{error_context}. Please generate a valid corrected Cypher statement."
        )
    else:
        prompt_question = question

    try:
        raw_cypher = enterprise_cypher_chain.cypher_generation_chain.invoke(
            {
                "schema": enterprise_cypher_chain.graph_schema,
                "question": prompt_question,
            }
        )
        query = extract_cypher(raw_cypher).strip()
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Generated Cypher query",
            extra={
                "event": "cypher_generation",
                "cypher_query": query,
                "status": "success",
                "execution_time_ms": duration_ms,
            },
        )
        return {
            "cypher_query": query,
            "cypher_error": None,
        }
    except Exception as e:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            "Cypher generation failed",
            extra={
                "event": "cypher_generation",
                "status": "failure",
                "execution_time_ms": duration_ms,
                "error": str(e),
            },
        )
        return {
            "cypher_query": "",
            "cypher_error": str(e),
        }


def validator(state: AgentState) -> Dict[str, Any]:
    """Runs CypherQueryCorrector and validates/executes query, retrying once on failure."""
    cypher_query = state.get("cypher_query", "")
    corrector = enterprise_cypher_chain.cypher_query_corrector
    retry_count = state.get("cypher_retry_count", 0)

    try:
        if not cypher_query:
            raise ValueError("No Cypher query was generated to validate.")

        # 1. Apply query guard: reject mutation keywords and ensure LIMIT
        guarded_query = apply_query_guard(
            cypher_query, default_limit=enterprise_cypher_chain.top_k or 200
        )

        # 2. Validate relationship direction against schema (ExtendedCypherQueryCorrector rejects reversed paths)
        corrected_query = corrector(guarded_query) if corrector else guarded_query
        if not corrected_query:
            raise ValueError(f"Cypher query corrector failed for: {guarded_query}")

        # 3. Execute query strictly in read-only session mode
        start_exec = time.perf_counter()
        context = execute_read_only_cypher(
            enterprise_cypher_chain.graph, corrected_query
        )[: enterprise_cypher_chain.top_k]

        if enterprise_cypher_chain.node_properties_to_exclude and isinstance(context, list):
            context = remove_keys_from_dicts(
                context, enterprise_cypher_chain.node_properties_to_exclude
            )

        # 4. Cache validated & executed Cypher query in Redis
        set_cached_cypher(state["input"], corrected_query)

        duration_ms = round((time.perf_counter() - start_exec) * 1000, 2)
        logger.info(
            "Validated and executed Cypher",
            extra={
                "event": "cypher_execution",
                "cypher_query": corrected_query,
                "status": "success",
                "execution_time_ms": duration_ms,
                "result_count": len(context) if isinstance(context, list) else 0,
            },
        )

        return {
            "cypher_query": corrected_query,
            "cypher_context": context,
            "cypher_error": None,
            "validation_passed": True,
        }
    except Exception as e:
        logger.warning(
            f"Cypher validation/execution error on attempt {retry_count + 1}: {e}",
            extra={
                "event": "cypher_execution",
                "cypher_query": cypher_query,
                "status": "failure",
                "error": str(e),
                "retry_count": retry_count,
            },
        )
        return {
            "cypher_error": str(e),
            "cypher_retry_count": retry_count + 1,
            "validation_passed": False,
        }


def should_retry_cypher(state: AgentState) -> str:
    """Conditional edge from validator: retry once on failure, else respond."""
    if state.get("validation_passed", False):
        return "respond"
    if state.get("cypher_retry_count", 0) <= 1:
        return "cypher_gen"
    return "respond"


def review_search(state: AgentState) -> Dict[str, Any]:
    """Executes vector search review chain."""
    question = state["input"]
    start_time = time.perf_counter()
    result = reviews_vector_chain.invoke(question)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    answer = result.get("result", str(result)) if isinstance(result, dict) else str(result)
    logger.info(
        "Vector review search node finished",
        extra={
            "event": "agent_tool_execution",
            "tool_name": "explore_patient_experiences",
            "status": "success",
            "execution_time_ms": duration_ms,
        },
    )

    step = (
        AgentActionStep(tool="explore_patient_experiences", tool_input=question),
        answer,
    )
    return {
        "tool_output": answer,
        "intermediate_steps": [step],
    }


def node_get_hospital_wait_time(state: AgentState) -> Dict[str, Any]:
    """Calculates wait time for a specific hospital."""
    question = state["input"]
    hospital = state.get("extracted_hospital") or question
    cleaned = hospital.replace("hospital", "").replace("Hospital", "").strip()
    result = get_current_wait_times(cleaned)
    step = (
        AgentActionStep(tool="get_hospital_wait_time", tool_input=cleaned),
        result,
    )
    return {
        "tool_output": result,
        "intermediate_steps": [step],
    }


def node_find_most_available_hospital(state: AgentState) -> Dict[str, Any]:
    """Finds the hospital with the shortest wait time."""
    question = state["input"]
    result = get_most_available_hospital(question)
    step = (
        AgentActionStep(tool="find_most_available_hospital", tool_input=question),
        result,
    )
    return {
        "tool_output": result,
        "intermediate_steps": [step],
    }


def respond(state: AgentState) -> Dict[str, Any]:
    """Formats final response and ensures consistent output payload."""
    route = state.get("route", "")
    question = state["input"]

    if route == "explore_hospital_database":
        context = state.get("cypher_context")
        if context is None and state.get("cypher_error"):
            output = (
                "I was unable to retrieve data from the database to answer your question. "
                "Please verify your question details or try rephrasing."
            )
        else:
            qa_res = enterprise_cypher_chain.qa_chain.invoke(
                {"question": question, "context": context if context is not None else []}
            )
            output = qa_res.get(enterprise_cypher_chain.qa_chain.output_key, str(qa_res))

        step = (
            AgentActionStep(
                tool="explore_hospital_database",
                tool_input=question,
                log=f"Generated Cypher: {state.get('cypher_query')}",
            ),
            output,
        )
        return {
            "output": output,
            "intermediate_steps": [step],
        }

    # Other paths (review_search, wait_time, availability) already recorded their output
    output = str(state.get("tool_output", ""))
    return {
        "output": output,
    }


# Standalone callable functions preserving existing tool behaviors
explore_hospital_database = enterprise_cypher_chain.invoke
explore_patient_experiences = reviews_vector_chain.invoke
get_hospital_wait_time = get_current_wait_times
find_most_available_hospital = get_most_available_hospital


def build_hospital_rag_graph() -> StateGraph:
    """Builds and wires the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add required nodes
    workflow.add_node("router", router)
    workflow.add_node("cypher_gen", cypher_gen)
    workflow.add_node("validator", validator)
    workflow.add_node("review_search", review_search)
    workflow.add_node("get_hospital_wait_time", node_get_hospital_wait_time)
    workflow.add_node("find_most_available_hospital", node_find_most_available_hospital)
    workflow.add_node("respond", respond)

    # Set router entry point
    workflow.set_entry_point("router")

    # Conditional routing from router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "explore_hospital_database": "cypher_gen",
            "explore_patient_experiences": "review_search",
            "get_hospital_wait_time": "get_hospital_wait_time",
            "find_most_available_hospital": "find_most_available_hospital",
        },
    )

    # Transition from cypher_gen to validator
    workflow.add_edge("cypher_gen", "validator")

    # Conditional edge from validator: retry once on failure, else respond
    workflow.add_conditional_edges(
        "validator",
        should_retry_cypher,
        {
            "cypher_gen": "cypher_gen",
            "respond": "respond",
        },
    )

    # Transitions to respond
    workflow.add_edge("review_search", "respond")
    workflow.add_edge("get_hospital_wait_time", "respond")
    workflow.add_edge("find_most_available_hospital", "respond")

    # Finish at respond
    workflow.add_edge("respond", END)

    return workflow


hospital_rag_graph = build_hospital_rag_graph()
graph_rag_agent_executor = hospital_rag_graph.compile()


async def astream_graph_rag_agent(
    question: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streams the hospital RAG agent execution and final answer tokens via SSE events."""
    state: AgentState = {
        "input": question,
        "output": "",
        "intermediate_steps": [],
        "route": "",
        "extracted_hospital": None,
        "cypher_query": None,
        "cypher_context": None,
        "cypher_error": None,
        "cypher_retry_count": 0,
        "validation_passed": False,
        "tool_output": None,
    }

    # 1. Route decision
    router_res = await asyncio.to_thread(router, state)
    state.update(router_res)
    route = state.get("route", "explore_hospital_database")
    yield {"type": "status", "stage": "route_selected", "route": route}

    # 2. Execute selected path and stream final answer tokens
    if route == "explore_hospital_database":
        while True:
            gen_res = await asyncio.to_thread(cypher_gen, state)
            state.update(gen_res)
            val_res = await asyncio.to_thread(validator, state)
            state.update(val_res)
            if state.get("validation_passed", False) or state.get("cypher_retry_count", 0) > 1:
                break

        if state.get("cypher_query"):
            yield {
                "type": "status",
                "stage": "cypher_executed",
                "cypher_query": state.get("cypher_query"),
            }

        context = state.get("cypher_context")
        final_output = ""
        if context is None and state.get("cypher_error"):
            final_output = (
                "I was unable to retrieve data from the database to answer your question. "
                "Please verify your question details or try rephrasing."
            )
            yield {"type": "token", "token": final_output}
        else:
            formatted_prompt = enterprise_cypher_chain.qa_chain.prompt.format(
                question=question, context=context if context is not None else []
            )
            async for chunk in enterprise_cypher_chain.qa_chain.llm.astream(formatted_prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    final_output += token
                    yield {"type": "token", "token": token}

        step = (
            AgentActionStep(
                tool="explore_hospital_database",
                tool_input=question,
                log=f"Generated Cypher: {state.get('cypher_query')}",
            ),
            final_output,
        )
        yield {
            "type": "done",
            "output": final_output,
            "intermediate_steps": [str(step)],
        }

    elif route == "explore_patient_experiences":
        yield {"type": "status", "stage": "vector_search"}
        docs = await asyncio.to_thread(
            reviews_vector_chain.retriever.get_relevant_documents, question
        )
        context_str = "\n\n".join(doc.page_content for doc in docs)
        messages = review_prompt.format_messages(
            context=context_str, question=question
        )

        final_output = ""
        async for chunk in reviews_vector_chain.combine_documents_chain.llm_chain.llm.astream(
            messages
        ):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                final_output += token
                yield {"type": "token", "token": token}

        step = (
            AgentActionStep(tool="explore_patient_experiences", tool_input=question),
            final_output,
        )
        yield {
            "type": "done",
            "output": final_output,
            "intermediate_steps": [str(step)],
        }

    elif route == "get_hospital_wait_time":
        hospital = state.get("extracted_hospital") or question
        cleaned = hospital.replace("hospital", "").replace("Hospital", "").strip()
        result = await asyncio.to_thread(get_current_wait_times, cleaned)
        yield {"type": "token", "token": result}
        step = (
            AgentActionStep(tool="get_hospital_wait_time", tool_input=cleaned),
            result,
        )
        yield {
            "type": "done",
            "output": result,
            "intermediate_steps": [str(step)],
        }

    elif route == "find_most_available_hospital":
        result = await asyncio.to_thread(get_most_available_hospital, question)
        yield {"type": "token", "token": result}
        step = (
            AgentActionStep(tool="find_most_available_hospital", tool_input=question),
            result,
        )
        yield {
            "type": "done",
            "output": result,
            "intermediate_steps": [str(step)],
        }

