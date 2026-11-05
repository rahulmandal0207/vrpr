"""Configure logging for the command line, as text or as JSON lines."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from vrpr.config import LoggingConfig

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(config: LoggingConfig) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if config.format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    logging.basicConfig(level=config.level, handlers=[handler], force=True)
