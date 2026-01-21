"""
Simplified logging system for Streamlit compatibility.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter for colored console output."""
    
    LOG_COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        level_color = self.LOG_COLORS.get(record.levelname, self.LOG_COLORS['RESET'])
        reset_color = self.LOG_COLORS['RESET']
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        message = f"{timestamp} | {level_color}{record.levelname:8}{reset_color} | {record.name:20} | {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class PipelineLogger:
    """
    Simple logger for pipeline components.
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
            self.setup_logger(name)
    
    def setup_logger(self, name: str, log_file: Optional[str] = None):
        """Setup logger with handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredConsoleFormatter())
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra={'component': self.component, **kwargs})
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra={'component': self.component, **kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra={'component': self.component, **kwargs})
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, extra={'component': self.component, **kwargs})
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra={'component': self.component, **kwargs})


class LoggingContext:
    """
    Context manager for logging with specific context.
    """
    
    def __init__(self, logger: PipelineLogger, operation: str, **context):
        """
        Initialize logging context.
        
        Args:
            logger: PipelineLogger instance
            operation: Operation name
            **context: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"Operation failed: {self.operation}",
                exc_info=(exc_type, exc_val, exc_tb),
                duration=duration,
                status="failed",
                **self.context
            )
        else:
            self.logger.info(
                f"Completed operation: {self.operation}",
                duration=duration,
                status="success",
                **self.context
            )


# Global logger instances for easy access
class GlobalLoggers:
    """Container for global logger instances."""
    
    def __init__(self):
        self._loggers = {}
    
    def get_logger(self, component: str) -> PipelineLogger:
        """Get or create logger for component."""
        if component not in self._loggers:
            self._loggers[component] = PipelineLogger(f"pipeline.{component}", component)
        return self._loggers[component]
    
    def scraper(self) -> PipelineLogger:
        return self.get_logger("scraper")
    
    def preprocessing(self) -> PipelineLogger:
        return self.get_logger("preprocessing")
    
    def models(self) -> PipelineLogger:
        return self.get_logger("models")
    
    def regression(self) -> PipelineLogger:
        return self.get_logger("regression")
    
    def xgboost(self) -> PipelineLogger:
        return self.get_logger("xgboost")
    
    def neural_network(self) -> PipelineLogger:
        return self.get_logger("neural_network")
    
    def knn(self) -> PipelineLogger:
        return self.get_logger("knn")
    
    def isolation_forest(self) -> PipelineLogger:
        return self.get_logger("isolation_forest")
    
    def explainability(self) -> PipelineLogger:
        return self.get_logger("explainability")
    
    def dashboard(self) -> PipelineLogger:
        return self.get_logger("dashboard")
    
    def pipeline(self) -> PipelineLogger:
        return self.get_logger("pipeline")
    
    def eda(self) -> PipelineLogger:
        return self.get_logger("eda")


# Global instance
loggers = GlobalLoggers()


# Quick access functions
def get_scraper_logger() -> PipelineLogger:
    return loggers.scraper()

def get_preprocessing_logger() -> PipelineLogger:
    return loggers.preprocessing()

def get_model_logger(model_type: str = None) -> PipelineLogger:
    if model_type:
        if hasattr(loggers, model_type):
            return getattr(loggers, model_type)()
    return loggers.models()

def get_dashboard_logger() -> PipelineLogger:
    return loggers.dashboard()

def get_pipeline_logger() -> PipelineLogger:
    return loggers.pipeline()


# Initialize basic logging
def initialize_logging():
    """Initialize basic logging system."""
    # Basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Log initialization
    logger = get_pipeline_logger()
    logger.info("Logging system initialized")


# Auto-initialize logging
initialize_logging()
