# app/logging_config.py
import logging
import json
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path("logs", "app.log")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """
    Configure root logging once for the entire application.

    - JSON logs
    - File + console output
    - Safe with uvicorn --reload
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on reload
    if any(isinstance(getattr(h, "formatter", None), JsonFormatter) for h in root_logger.handlers):
        return

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)

    # Optional local/dev file sink; keep stdout as primary transport for deployments.
    if _env_bool("ENABLE_FILE_LOGGING", default=False):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_PATH,
            maxBytes=25 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    # Ensure uvicorn logs flow through root handlers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).propagate = True
