"""Structured logging for Aegis backend."""

import logging
import os
from datetime import datetime
from typing import Any, Dict
import json


class StructuredLogger:
    """Structured logging with JSON output for each pipeline stage."""
    
    def __init__(self, name: str, log_dir: str = "./logs", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler - one per process
        log_file = os.path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, level))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level))
        
        # JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_event(
        self, 
        event_name: str, 
        stage: str,
        data: Dict[str, Any] = None,
        level: str = "INFO"
    ):
        """Log structured event with context."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_name,
            "stage": stage,
            "data": data or {}
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)


# Global logger instances
radar_logger = StructuredLogger("radar", level="INFO")
detection_logger = StructuredLogger("detection", level="INFO")
tracking_logger = StructuredLogger("tracking", level="INFO")
threat_logger = StructuredLogger("threat", level="INFO")
ew_logger = StructuredLogger("ew", level="INFO")
pipeline_logger = StructuredLogger("pipeline", level="INFO")
websocket_logger = StructuredLogger("websocket", level="INFO")
