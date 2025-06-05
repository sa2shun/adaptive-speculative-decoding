"""
Exception classes for adaptive speculative decoding.

This module defines all custom exceptions used throughout the system
with proper error handling and debugging information.
"""

from typing import Optional, Dict, Any
import traceback
import time

class AdaptiveDecodingError(Exception):
    """Base exception for all adaptive speculative decoding errors."""
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        self.timestamp = time.time()
        self.traceback_str = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'cause': str(self.cause) if self.cause else None,
            'traceback': self.traceback_str
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        parts = [f"{self.error_code}: {self.message}"]
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        
        return " | ".join(parts)

class ConfigurationError(AdaptiveDecodingError):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None, **kwargs):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = config_value
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details=details,
            **kwargs
        )

class ModelLoadError(AdaptiveDecodingError):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 model_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if model_path:
            details['model_path'] = model_path
        
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details=details,
            **kwargs
        )

class PredictionError(AdaptiveDecodingError):
    """Raised when quality prediction fails."""
    
    def __init__(self, message: str, predictor_type: Optional[str] = None,
                 input_features: Optional[Dict] = None, **kwargs):
        details = kwargs.get('details', {})
        if predictor_type:
            details['predictor_type'] = predictor_type
        if input_features:
            details['input_features'] = str(input_features)  # Convert to string for serialization
        
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details,
            **kwargs
        )

class OptimizationError(AdaptiveDecodingError):
    """Raised when optimization algorithm fails."""
    
    def __init__(self, message: str, algorithm: Optional[str] = None,
                 system_state: Optional[Dict] = None, **kwargs):
        details = kwargs.get('details', {})
        if algorithm:
            details['algorithm'] = algorithm
        if system_state:
            details['system_state'] = str(system_state)
        
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_ERROR",
            details=details,
            **kwargs
        )

class QualityEvaluationError(AdaptiveDecodingError):
    """Raised when quality evaluation fails."""
    
    def __init__(self, message: str, metric_type: Optional[str] = None,
                 output_text: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if metric_type:
            details['metric_type'] = metric_type
        if output_text:
            # Truncate long output for logging
            details['output_text'] = output_text[:200] + "..." if len(output_text) > 200 else output_text
        
        super().__init__(
            message=message,
            error_code="QUALITY_EVAL_ERROR",
            details=details,
            **kwargs
        )

class ResourceError(AdaptiveDecodingError):
    """Raised when resource allocation or management fails."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 required_amount: Optional[float] = None,
                 available_amount: Optional[float] = None, **kwargs):
        details = kwargs.get('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if required_amount is not None:
            details['required_amount'] = required_amount
        if available_amount is not None:
            details['available_amount'] = available_amount
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details=details,
            **kwargs
        )

class TimeoutError(AdaptiveDecodingError):
    """Raised when operations exceed timeout limits."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if timeout_seconds is not None:
            details['timeout_seconds'] = timeout_seconds
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=details,
            **kwargs
        )

class ValidationError(AdaptiveDecodingError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 field_value: Optional[Any] = None, 
                 validation_rule: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)
        if validation_rule:
            details['validation_rule'] = validation_rule
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            **kwargs
        )

class CacheError(AdaptiveDecodingError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_type: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if cache_type:
            details['cache_type'] = cache_type
        if operation:
            details['operation'] = operation
        
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
            **kwargs
        )

class NetworkError(AdaptiveDecodingError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None,
                 status_code: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if endpoint:
            details['endpoint'] = endpoint
        if status_code is not None:
            details['status_code'] = status_code
        
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            details=details,
            **kwargs
        )

# Exception utilities
class ExceptionHandler:
    """Utility class for handling exceptions in a consistent manner."""
    
    @staticmethod
    def log_exception(exception: Exception, logger, context: Optional[Dict] = None):
        """Log an exception with proper formatting."""
        if isinstance(exception, AdaptiveDecodingError):
            error_dict = exception.to_dict()
            if context:
                error_dict['context'] = context
            logger.error(f"AdaptiveDecodingError: {error_dict}")
        else:
            logger.error(f"Unexpected error: {exception}", exc_info=True)
    
    @staticmethod
    def wrap_exception(original_exception: Exception, 
                      new_exception_class: type = AdaptiveDecodingError,
                      message: Optional[str] = None,
                      **kwargs) -> AdaptiveDecodingError:
        """Wrap an exception in an AdaptiveDecodingError."""
        if message is None:
            message = f"Wrapped exception: {original_exception}"
        
        return new_exception_class(
            message=message,
            cause=original_exception,
            **kwargs
        )
    
    @staticmethod
    def reraise_with_context(exception: Exception, context: Dict[str, Any]):
        """Re-raise an exception with additional context."""
        if isinstance(exception, AdaptiveDecodingError):
            exception.details.update(context)
            raise exception
        else:
            raise AdaptiveDecodingError(
                message=f"Exception with context: {exception}",
                cause=exception,
                details=context
            )

# Decorator for automatic exception handling
def handle_exceptions(exception_type: type = AdaptiveDecodingError,
                     message: Optional[str] = None,
                     log_errors: bool = True):
    """Decorator to automatically handle and wrap exceptions."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AdaptiveDecodingError:
                # Re-raise our custom exceptions as-is
                raise
            except Exception as e:
                if log_errors:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    ExceptionHandler.log_exception(e, logger, {
                        'function': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs)
                    })
                
                wrapped_message = message or f"Error in {func.__name__}: {e}"
                raise ExceptionHandler.wrap_exception(
                    e, exception_type, wrapped_message
                )
        
        return wrapper
    return decorator