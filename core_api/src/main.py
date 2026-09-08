import json
import logging
import time
from typing import Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from src.utils.logging_config import configure_logging
from src.agents.graph_rag_agent import (
    astream_graph_rag_agent,
    graph_rag_agent_executor,
)
from src.models.enterprise_rag_query import EnterpriseQueryInput, HospitalQueryOutput
from src.utils.async_utils import async_retry

configure_logging()
logger = logging.getLogger("core_api")

app = FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await graph_rag_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/graph-rag-agent", response_model=Union[HospitalQueryOutput, None])
async def ask_hospital_agent(
    query: EnterpriseQueryInput,
    request: Request,
):
    accept_header = request.headers.get("accept", "")
    wants_stream = query.stream or "text/event-stream" in accept_header

    if wants_stream:
        logger.info(
            "Streaming request received",
            extra={
                "event": "agent_request_start",
                "stream": True,
                "query": query.text,
            },
        )

        async def sse_event_generator():
            start_time = time.perf_counter()
            try:
                async for event in astream_graph_rag_agent(query.text):
                    yield f"data: {json.dumps(event)}\n\n"
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.info(
                    "Streaming request completed successfully",
                    extra={
                        "event": "agent_request_completed",
                        "status": "success",
                        "stream": True,
                        "execution_time_ms": duration_ms,
                        "query": query.text,
                    },
                )
            except Exception as e:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.error(
                    "Streaming request failed",
                    extra={
                        "event": "agent_request_completed",
                        "status": "failure",
                        "stream": True,
                        "execution_time_ms": duration_ms,
                        "query": query.text,
                        "error": str(e),
                    },
                )
                err_event = {
                    "type": "done",
                    "output": f"An error occurred while streaming response: {e}",
                    "intermediate_steps": [],
                }
                yield f"data: {json.dumps(err_event)}\n\n"

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming fallback path (used by admin_query_portal and existing sync tests)
    start_time = time.perf_counter()
    logger.info(
        "Request received",
        extra={
            "event": "agent_request_start",
            "stream": False,
            "query": query.text,
        },
    )
    try:
        query_response = await invoke_agent_with_retry(query.text)
        query_response["intermediate_steps"] = [
            str(s) for s in query_response["intermediate_steps"]
        ]
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Request completed successfully",
            extra={
                "event": "agent_request_completed",
                "status": "success",
                "stream": False,
                "execution_time_ms": duration_ms,
                "query": query.text,
            },
        )
        return HospitalQueryOutput(
            input=query.text,
            output=query_response["output"],
            intermediate_steps=query_response["intermediate_steps"],
        )
    except Exception as e:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(
            "Request failed",
            extra={
                "event": "agent_request_completed",
                "status": "failure",
                "execution_time_ms": duration_ms,
                "query": query.text,
                "error": str(e),
            },
        )
        raise
