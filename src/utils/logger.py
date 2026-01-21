"""
Comprehensive logging system for Market Narrative Risk Intelligence System.
Provides structured, professional logging for all pipeline components.
"""
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import inspect
import threading
from logging.handlers import RotatingFileHandler


class StructuredFormatter(logging.Formatter):
    """Structured log formatter for machine-readable output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Professional console formatter with clean output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        # Get caller information
        caller_info = self._get_caller_info()
        
        # Color the level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Create formatted message
        formatted_message = (
            f"{timestamp} | "
            f"{level_color}{record.levelname:8}{reset_color} | "
            f"{caller_info:25} | "
            f"{record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted_message += f"\n{self.formatException(record.exc_info)}"
        
        return formatted_message
    
    def _get_caller_info(self) -> str:
        """Get information about the caller for logging."""
        try:
            # Walk up the stack to find the first non-logging caller
            stack = inspect.stack()
            for frame_info in stack:
                frame = frame_info.frame
                module_name = frame.f_globals.get('__name__', '')
                if module_name != 'logging' and not module_name.startswith('src.utils.logger'):
                    filename = Path(frame_info.filename).name
                    return f"{filename}:{frame_info.lineno}"
        except:
            pass
        return "unknown:0"


class PipelineLogger:
    """
    Professional logger for pipeline components with structured context.
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
        self.context = {
            'component': self.component,
            'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name
        }
    
    def add_context(self, **kwargs):
        """Add contextual information to log messages."""
        self.context.update(kwargs)
    
    def _log_with_context(self, level: int, message: str, 
                         extra: Dict = None, exc_info: bool = False):
        """Log message with context and structured formatting."""
        # Combine context with any extra fields
        log_extra = self.context.copy()
        if extra:
            log_extra.update(extra)
        
        # Ensure logger has handlers
        if not self.logger.handlers:
            self._setup_default_handlers()
        
        self.logger.log(level, message, extra={'extra': log_extra}, exc_info=exc_info)
    
    def _setup_default_handlers(self):
        """Set up default handlers for this logger."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler for component-specific logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_dir / f"{self.component}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log_with_context(logging.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log_with_context(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log_with_context(logging.WARNING, message, kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error level message."""
        self._log_with_context(logging.ERROR, message, kwargs, exc_info)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self._log_with_context(logging.CRITICAL, message, kwargs)
    
    def metric(self, name: str, value: float, **kwargs):
        """Log performance metric."""
        metric_data = {'metric_name': name, 'metric_value': value, **kwargs}
        self._log_with_context(
            logging.INFO, 
            f"Metric recorded: {name}={value}", 
            metric_data
        )
    
    def timing(self, operation: str, duration: float, **kwargs):
        """Log timing information."""
        timing_data = {
            'operation': operation, 
            'duration_ms': duration * 1000, 
            **kwargs
        }
        self._log_with_context(
            logging.INFO, 
            f"Operation timing: {operation} completed in {duration:.3f} seconds", 
            timing_data
        )
    
    def event(self, event_type: str, details: str = "", **kwargs):
        """Log system event."""
        event_data = {
            'event_type': event_type, 
            'event_details': details, 
            **kwargs
        }
        self._log_with_context(
            logging.INFO, 
            f"System event: {event_type} - {details}", 
            event_data
        )
    
    def data_point(self, metric: str, value: Any, **kwargs):
        """Log data point for monitoring."""
        data = {'data_metric': metric, 'data_value': value, **kwargs}
        self._log_with_context(
            logging.INFO, 
            f"Data point: {metric} = {value}", 
            data
        )


class LoggingContext:
    """
    Context manager for logging operations with timing and status tracking.
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
        """Enter context and log operation start."""
        self.start_time = datetime.now()
        self.logger.add_context(operation=self.operation, **self.context)
        self.logger.info(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and log operation completion."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"Operation failed: {self.operation}",
                exc_info=(exc_type, exc_val, exc_tb),
                duration=duration,
                status="failure"
            )
        else:
            self.logger.info(
                f"Operation completed: {self.operation}",
                duration=duration,
                status="success"
            )


def setup_component_logger(
    component: str,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> PipelineLogger:
    """
    Set up a logger for a specific component.
    
    Args:
        component: Component name
        log_level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    
    Returns:
        Configured PipelineLogger instance
    """
    # Get numeric log level
    level = getattr(logging, log_level.upper())
    
    # Create logger
    logger_name = f"market_risk.{component}"
    logger = PipelineLogger(logger_name, component)
    
    # Clear existing handlers
    logger.logger.handlers.clear()
    
    # Set log level
    logger.logger.setLevel(level)
    logger.logger.propagate = False
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.setLevel(level)
        logger.logger.addHandler(console_handler)
    
    # Add file handler
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_dir / f"{component}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(level)
        logger.logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized for component: {component}")
    
    return logger


class LoggingRegistry:
    """
    Registry for managing all component loggers.
    """
    
    def __init__(self):
        self._loggers: Dict[str, PipelineLogger] = {}
        self._lock = threading.Lock()
    
    def get_logger(self, component: str) -> PipelineLogger:
        """Get or create logger for component."""
        with self._lock:
            if component not in self._loggers:
                # Get logging configuration
                from src.utils.config_loader import get_value
                log_level = get_value(f'logging.component_levels.{component}', 'INFO')
                log_to_file = get_value('logging.file_output', True)
                log_to_console = get_value('logging.console_output', True)
                
                self._loggers[component] = setup_component_logger(
                    component=component,
                    log_level=log_level,
                    log_to_file=log_to_file,
                    log_to_console=log_to_console
                )
            
            return self._loggers[component]
    
    def get_all_loggers(self) -> Dict[str, PipelineLogger]:
        """Get all registered loggers."""
        return self._loggers.copy()
    
    def set_level(self, component: str, level: str):
        """Set log level for a component."""
        logger = self.get_logger(component)
        numeric_level = getattr(logging, level.upper())
        logger.logger.setLevel(numeric_level)
        
        # Update all handlers
        for handler in logger.logger.handlers:
            handler.setLevel(numeric_level)
    
    def flush_all(self):
        """Flush all log handlers."""
        for logger in self._loggers.values():
            for handler in logger.logger.handlers:
                handler.flush()


# Global logging registry
_logging_registry = LoggingRegistry()


def get_logger(component: str) -> PipelineLogger:
    """
    Get logger for specified component.
    
    Args:
        component: Component name
    
    Returns:
        PipelineLogger instance
    """
    return _logging_registry.get_logger(component)


# Convenience functions for common components
def get_scraper_logger() -> PipelineLogger:
    return get_logger("scraper")

def get_preprocessing_logger() -> PipelineLogger:
    return get_logger("preprocessing")

def get_feature_engineering_logger() -> PipelineLogger:
    return get_logger("feature_engineering")

def get_model_logger(model_type: str = None) -> PipelineLogger:
    if model_type:
        return get_logger(f"model.{model_type}")
    return get_logger("models")

def get_dashboard_logger() -> PipelineLogger:
    return get_logger("dashboard")

def get_pipeline_logger() -> PipelineLogger:
    return get_logger("pipeline")

def get_explainability_logger() -> PipelineLogger:
    return get_logger("explainability")

def get_eda_logger() -> PipelineLogger:
    return get_logger("eda")


def initialize_system_logging():
    """
    Initialize comprehensive logging system for the entire application.
    """
    from src.utils.config_loader import get_value
    
    # Get logging configuration
    default_level = get_value('logging.level', 'INFO')
    log_dir = get_value('logging.directory', 'logs')
    json_format = get_value('logging.json_format', False)
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Only capture warnings and above
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create main application logger
    main_logger = get_pipeline_logger()
    
    # Log initialization
    main_logger.info("Logging system initialized", extra={
        'log_level': default_level,
        'log_directory': str(log_path.absolute()),
        'json_format': json_format
    })
    
    return main_logger


# Logging configuration helper
def configure_from_dict(config: Dict[str, Any]):
    """
    Configure logging from dictionary.
    
    Args:
        config: Configuration dictionary
    """
    # Set component levels
    component_levels = config.get('component_levels', {})
    for component, level in component_levels.items():
        _logging_registry.set_level(component, level)
    
    # Log configuration
    logger = get_pipeline_logger()
    logger.info("Logging configuration applied", extra={'config': config})


# Performance monitoring decorator
def log_performance(operation_name: str = None):
    """
    Decorator to log performance of functions.
    
    Args:
        operation_name: Name of operation (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Determine component from module
            module_name = func.__module__
            component = module_name.split('.')[-1] if '.' in module_name else module_name
            
            logger = get_logger(component)
            op_name = operation_name or func.__name__
            
            with LoggingContext(logger, op_name):
                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.timing(op_name, duration)
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.error(
                        f"Operation {op_name} failed after {duration:.3f}s",
                        exc_info=True
                    )
                    raise
        
        return wrapper
    return decorator


# Initialize logging when module is imported
try:
    # Initialize only if not already initialized
    if not _logging_registry.get_all_loggers():
        initialize_system_logging()
except Exception as e:
    # Fallback to basic logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger(__name__).warning(
        "Could not initialize advanced logging system", 
        exc_info=True
    )
