# app/logging_config.py
import logging
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path("logs", "app.log")


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
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on reload
    if any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        return

    # file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler = RotatingFileHandler(      # ensure the file doesn't explose in size, by renaming on max size reached. Oldest files are deleted automatically
                    LOG_PATH,
                    maxBytes=25 * 1024 * 1024,  # 25 MB
                    backupCount=5               # keep last 5 files
                )      
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)

    # Ensure uvicorn logs flow through root handlers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(logger_name).propagate = True
