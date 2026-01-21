"""
Logging utility for the entire pipeline.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class PipelineLogger:
    """
    Pipeline-specific logger with contextual information.
    """
    
    def __init__(self, component: str):
        """
        Initialize pipeline logger.
        
        Args:
            component: Component name (scraper, model, etc.)
        """
        self.component = component
        self.logger = setup_logger(
            name=f"pipeline.{component}",
            log_file=f"logs/{component}.log"
        )
    
    def info(self, message: str):
        """Log info message with component context."""
        self.logger.info(f"[{self.component}] {message}")
    
    def warning(self, message: str):
        """Log warning message with component context."""
        self.logger.warning(f"[{self.component}] {message}")
    
    def error(self, message: str, exc_info: bool = False):
        """Log error message with component context."""
        self.logger.error(f"[{self.component}] {message}", exc_info=exc_info)
    
    def debug(self, message: str):
        """Log debug message with component context."""
        self.logger.debug(f"[{self.component}] {message}")


# Global logger instances
scraper_logger = PipelineLogger("scraper")
preprocessing_logger = PipelineLogger("preprocessing")
model_logger = PipelineLogger("model")
dashboard_logger = PipelineLogger("dashboard")
