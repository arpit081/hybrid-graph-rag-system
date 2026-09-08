import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

STANDARD_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
}


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON strings with custom structured attributes."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom structured logging fields passed in extra={...}
        for key, value in record.__dict__.items():
            if key not in STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, OverflowError):
                    log_data[key] = str(value)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(level: int = logging.INFO) -> None:
    """Configures the root logger to use JSONFormatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]


def wrap_vector_store_with_logging(vector_store: Any, index_name: str) -> Any:
    """Wraps similarity_search on a vector store to log query, execution time, and status."""
    orig_similarity_search = vector_store.similarity_search
    logger = logging.getLogger("vector_search")

    def logged_similarity_search(query: str, *args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            results = orig_similarity_search(query, *args, **kwargs)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                "Vector search completed",
                extra={
                    "event": "vector_search",
                    "index_name": index_name,
                    "query": query,
                    "status": "success",
                    "execution_time_ms": duration_ms,
                    "result_count": len(results) if isinstance(results, list) else 0,
                },
            )
            return results
        except Exception as e:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "Vector search failed",
                extra={
                    "event": "vector_search",
                    "index_name": index_name,
                    "query": query,
                    "status": "failure",
                    "execution_time_ms": duration_ms,
                    "error": str(e),
                },
            )
            raise

    vector_store.similarity_search = logged_similarity_search
    return vector_store
