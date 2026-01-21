"""
Simplified logging module without circular imports.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        console: Whether to add console handler
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger


class PipelineLogger:
    """
    Simple pipeline logger for components.
    """
    
    def __init__(self, name: str, component: str = None):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            component: Component name
        """
        self.logger = logging.getLogger(name)
        self.component = component or name.split('.')[-1]
        
        # Setup logger if no handlers
        if not self.logger.handlers:
            # Default setup - can be overridden later
            setup_logger(name, console=True)
    
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


# Create global logger instances without importing config_loader
_scraper_logger = None
_preprocessing_logger = None
_model_logger = None
_dashboard_logger = None
_pipeline_logger = None


def get_scraper_logger() -> PipelineLogger:
    """Get scraper logger."""
    global _scraper_logger
    if _scraper_logger is None:
        _scraper_logger = PipelineLogger("pipeline.scraper", "scraper")
    return _scraper_logger


def get_preprocessing_logger() -> PipelineLogger:
    """Get preprocessing logger."""
    global _preprocessing_logger
    if _preprocessing_logger is None:
        _preprocessing_logger = PipelineLogger("pipeline.preprocessing", "preprocessing")
    return _preprocessing_logger


def get_model_logger() -> PipelineLogger:
    """Get model logger."""
    global _model_logger
    if _model_logger is None:
        _model_logger = PipelineLogger("pipeline.models", "models")
    return _model_logger


def get_dashboard_logger() -> PipelineLogger:
    """Get dashboard logger."""
    global _dashboard_logger
    if _dashboard_logger is None:
        _dashboard_logger = PipelineLogger("pipeline.dashboard", "dashboard")
    return _dashboard_logger


def get_pipeline_logger() -> PipelineLogger:
    """Get pipeline logger."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = PipelineLogger("pipeline.main", "pipeline")
    return _pipeline_logger


def setup_pipeline_logging(log_dir: str = "logs", level: str = "INFO"):
    """
    Set up logging for the entire pipeline.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    log_level = getattr(logging, level.upper())
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Setup component loggers
    components = {
        "scraper": get_scraper_logger(),
        "preprocessing": get_preprocessing_logger(),
        "models": get_model_logger(),
        "dashboard": get_dashboard_logger(),
        "pipeline": get_pipeline_logger()
    }
    
    for component_name, logger in components.items():
        log_file = log_dir_path / f"{component_name}.log"
        
        # Remove existing handlers
        logger.logger.handlers.clear()
        
        # Setup with file and console
        setup_logger(
            name=logger.logger.name,
            log_file=str(log_file),
            level=log_level,
            console=True
        )
        
        logger.info(f"Logger initialized for {component_name}")


# Set up basic logging on import
setup_pipeline_logging()
