"""
Logging utilities for consistent logging throughout the system.
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if requested
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'message', 'exc_info', 'exc_text', 'stack_info'}:
                    log_data[key] = value
        
        return json.dumps(log_data)

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )
        
        return super().format(record)

def setup_logging(config: Dict[str, Any], log_dir: Optional[str] = None) -> None:
    """
    Setup logging configuration for the entire application.
    
    Args:
        config: Logging configuration dictionary
        log_dir: Directory for log files (optional)
    """
    
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get('root_level', 'INFO')))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if config.get('use_structured_logging', False):
        console_formatter = StructuredFormatter()
    else:
        console_format = config.get('console_format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if config.get('use_colors', True) and sys.stdout.isatty():
            console_formatter = ColoredFormatter(console_format)
        else:
            console_formatter = logging.Formatter(console_format)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if log directory is specified
    if log_dir:
        file_format = config.get('file_format',
                                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        
        # Main log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'adaptive_sd.log'),
            maxBytes=config.get('max_file_size_mb', 100) * 1024 * 1024,
            backupCount=config.get('backup_count', 5)
        )
        
        if config.get('use_structured_logging', False):
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(file_format)
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, 'error.log'),
            maxBytes=config.get('max_file_size_mb', 100) * 1024 * 1024,
            backupCount=config.get('backup_count', 5)
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    # Set levels for specific modules
    module_levels = config.get('module_levels', {})
    for module_name, level in module_levels.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, level))
    
    # Log the logging setup
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.debug(f"Logging configuration: {config}")

def get_logger(name: str, 
               extra_fields: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional extra fields.
    
    Args:
        name: Logger name (usually __name__)
        extra_fields: Additional fields to include in all log messages
        
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Create a custom LoggerAdapter that includes extra fields
        class ExtraFieldsAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                kwargs.setdefault('extra', {}).update(self.extra)
                return msg, kwargs
        
        logger = ExtraFieldsAdapter(logger, extra_fields)
    
    return logger

class LoggerMixin:
    """Mixin class that provides logging capabilities to other classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
        self._logger_extra_fields = {}
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if self._logger is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = get_logger(logger_name, self._logger_extra_fields)
        return self._logger
    
    def set_logger_extra_fields(self, **fields) -> None:
        """Set extra fields to include in all log messages."""
        self._logger_extra_fields.update(fields)
        # Reset logger to pick up new fields
        self._logger = None
    
    def log_method_entry(self, method_name: str, **kwargs):
        """Log method entry with parameters."""
        self.logger.debug(f"Entering {method_name}", extra={
            'method': method_name,
            'parameters': kwargs
        })
    
    def log_method_exit(self, method_name: str, result: Any = None):
        """Log method exit with result."""
        self.logger.debug(f"Exiting {method_name}", extra={
            'method': method_name,
            'result_type': type(result).__name__ if result is not None else None
        })
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.logger.info(f"Performance: {operation}", extra={
            'operation': operation,
            'duration_seconds': duration,
            **metrics
        })

class ContextLogger:
    """Context manager for adding extra context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_handlers = []
    
    def __enter__(self):
        # Add context to all handlers
        for handler in self.logger.handlers:
            if hasattr(handler, 'formatter') and hasattr(handler.formatter, 'include_extra'):
                # Store original extra fields
                if not hasattr(handler, '_original_extra'):
                    handler._original_extra = {}
                
                # Add new context
                for key, value in self.context.items():
                    handler._original_extra[key] = getattr(handler, key, None)
                    setattr(handler, key, value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original extra fields
        for handler in self.logger.handlers:
            if hasattr(handler, '_original_extra'):
                for key, original_value in handler._original_extra.items():
                    if original_value is None:
                        delattr(handler, key)
                    else:
                        setattr(handler, key, original_value)
                delattr(handler, '_original_extra')

def log_function_call(include_args: bool = True, include_result: bool = False):
    """Decorator to automatically log function calls."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            # Log function entry
            if include_args:
                logger.debug(f"Calling {func.__name__}", extra={
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                })
            else:
                logger.debug(f"Calling {func.__name__}", extra={'function': func.__name__})
            
            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                log_data = {
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'status': 'success'
                }
                
                if include_result:
                    log_data['result_type'] = type(result).__name__
                
                logger.debug(f"Completed {func.__name__}", extra=log_data)
                
                return result
                
            except Exception as e:
                # Log exception
                logger.error(f"Exception in {func.__name__}: {e}", extra={
                    'function': func.__name__,
                    'exception_type': type(e).__name__,
                    'exception_message': str(e)
                })
                raise
        
        return wrapper
    return decorator

# Performance logging utilities
class PerformanceLogger:
    """Utility for tracking and logging performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.metrics[operation] = {'start_time': time.time()}
    
    def end_timer(self, operation: str, **extra_metrics) -> float:
        """End timing an operation and log the duration."""
        if operation not in self.metrics:
            self.logger.warning(f"Timer not started for operation: {operation}")
            return 0.0
        
        start_time = self.metrics[operation]['start_time']
        duration = time.time() - start_time
        
        self.logger.info(f"Performance: {operation}", extra={
            'operation': operation,
            'duration_seconds': duration,
            **extra_metrics
        })
        
        # Clean up
        del self.metrics[operation]
        
        return duration
    
    def log_metric(self, metric_name: str, value: float, **context):
        """Log a single metric value."""
        self.logger.info(f"Metric: {metric_name} = {value}", extra={
            'metric_name': metric_name,
            'metric_value': value,
            **context
        })