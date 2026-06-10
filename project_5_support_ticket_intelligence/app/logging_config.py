# app/logging_config.py
import json
import logging
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs logs as single-line JSON structures."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Include exception trace details if an error occurred
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Merge extra fields passed via extra={"extra_fields": {...}}
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
            
        return json.dumps(log_data)


def setup_logging():
    """Configures the root logger to use our JSON Formatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear default handlers to avoid duplicate log lines
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Redirect logs to standard output using the JSONFormatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)