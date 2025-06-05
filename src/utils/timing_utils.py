"""
Timing utilities for performance measurement and monitoring.
"""

import time
import asyncio
import functools
from typing import Optional, Dict, Any, Callable, Union
from contextlib import contextmanager, asynccontextmanager
import threading
from collections import defaultdict, deque
import statistics

class Timer:
    """High-precision timer for measuring execution time."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
        self.is_running = False
    
    def start(self) -> 'Timer':
        """Start the timer."""
        if self.is_running:
            raise RuntimeError(f"Timer '{self.name}' is already running")
        
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_time = None
        self.is_running = True
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if not self.is_running:
            raise RuntimeError(f"Timer '{self.name}' is not running")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        self.is_running = False
        return self.elapsed_time
    
    def reset(self) -> 'Timer':
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.is_running = False
        return self
    
    def restart(self) -> 'Timer':
        """Reset and start the timer."""
        return self.reset().start()
    
    @property
    def current_elapsed(self) -> float:
        """Get current elapsed time without stopping the timer."""
        if not self.is_running:
            return self.elapsed_time or 0.0
        return time.perf_counter() - self.start_time
    
    def __enter__(self) -> 'Timer':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __str__(self) -> str:
        """String representation."""
        if self.elapsed_time is not None:
            return f"Timer '{self.name}': {self.elapsed_time:.6f}s"
        elif self.is_running:
            return f"Timer '{self.name}': {self.current_elapsed:.6f}s (running)"
        else:
            return f"Timer '{self.name}': not started"

@contextmanager
def measure_time(name: str = "Operation"):
    """Context manager for measuring execution time."""
    timer = Timer(name)
    try:
        timer.start()
        yield timer
    finally:
        if timer.is_running:
            timer.stop()

@asynccontextmanager
async def async_measure_time(name: str = "AsyncOperation"):
    """Async context manager for measuring execution time."""
    timer = Timer(name)
    try:
        timer.start()
        yield timer
    finally:
        if timer.is_running:
            timer.stop()

def time_function(name: Optional[str] = None, 
                 logger: Optional[Any] = None,
                 include_args: bool = False):
    """Decorator to time function execution."""
    
    def decorator(func: Callable) -> Callable:
        timer_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with measure_time(timer_name) as timer:
                result = func(*args, **kwargs)
                
                if logger:
                    log_data = {
                        'function': func.__name__,
                        'duration_seconds': timer.elapsed_time
                    }
                    if include_args:
                        log_data['args_count'] = len(args)
                        log_data['kwargs_count'] = len(kwargs)
                    
                    logger.info(f"Function timing: {timer_name}", extra=log_data)
                
                return result
        
        return wrapper
    return decorator

def time_async_function(name: Optional[str] = None,
                       logger: Optional[Any] = None,
                       include_args: bool = False):
    """Decorator to time async function execution."""
    
    def decorator(func: Callable) -> Callable:
        timer_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with async_measure_time(timer_name) as timer:
                result = await func(*args, **kwargs)
                
                if logger:
                    log_data = {
                        'function': func.__name__,
                        'duration_seconds': timer.elapsed_time
                    }
                    if include_args:
                        log_data['args_count'] = len(args)
                        log_data['kwargs_count'] = len(kwargs)
                    
                    logger.info(f"Async function timing: {timer_name}", extra=log_data)
                
                return result
        
        return wrapper
    return decorator

class PerformanceProfiler:
    """Performance profiler for tracking multiple operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def start_timing(self, operation: str, operation_id: str = None) -> str:
        """Start timing an operation."""
        key = f"{operation}:{operation_id}" if operation_id else operation
        
        with self.lock:
            if key in self.active_timers:
                raise RuntimeError(f"Operation '{key}' is already being timed")
            
            self.active_timers[key] = time.perf_counter()
        
        return key
    
    def end_timing(self, key: str) -> float:
        """End timing an operation."""
        end_time = time.perf_counter()
        
        with self.lock:
            if key not in self.active_timers:
                raise RuntimeError(f"Operation '{key}' was not started")
            
            start_time = self.active_timers.pop(key)
            duration = end_time - start_time
            
            # Extract operation name (remove ID if present)
            operation = key.split(':')[0]
            self.timings[operation].append(duration)
        
        return duration
    
    @contextmanager
    def time_operation(self, operation: str, operation_id: str = None):
        """Context manager for timing operations."""
        key = self.start_timing(operation, operation_id)
        try:
            yield key
        finally:
            self.end_timing(key)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self.lock:
            timings = list(self.timings[operation])
        
        if not timings:
            return {}
        
        return {
            'count': len(timings),
            'total': sum(timings),
            'mean': statistics.mean(timings),
            'median': statistics.median(timings),
            'min': min(timings),
            'max': max(timings),
            'std': statistics.stdev(timings) if len(timings) > 1 else 0.0,
            'p95': statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings),
            'p99': statistics.quantiles(timings, n=100)[98] if len(timings) >= 100 else max(timings)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {operation: self.get_stats(operation) for operation in self.timings.keys()}
    
    def reset_stats(self, operation: Optional[str] = None) -> None:
        """Reset statistics for an operation or all operations."""
        with self.lock:
            if operation:
                self.timings[operation].clear()
            else:
                self.timings.clear()
    
    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their elapsed times."""
        current_time = time.perf_counter()
        with self.lock:
            return {
                key: current_time - start_time 
                for key, start_time in self.active_timers.items()
            }

class RateLimiter:
    """Rate limiter for controlling operation frequency."""
    
    def __init__(self, max_calls: int, time_window: float = 1.0):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to proceed with an operation.
        
        Args:
            timeout: Maximum time to wait for permission
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while True:
            current_time = time.time()
            
            with self.lock:
                # Remove old calls outside the time window
                while self.calls and current_time - self.calls[0] > self.time_window:
                    self.calls.popleft()
                
                # Check if we can proceed
                if len(self.calls) < self.max_calls:
                    self.calls.append(current_time)
                    return True
            
            # Check timeout
            if timeout is not None and (current_time - start_time) >= timeout:
                return False
            
            # Wait a bit before trying again
            time.sleep(0.01)
    
    @contextmanager
    def limit(self, timeout: Optional[float] = None):
        """Context manager for rate limiting."""
        if not self.acquire(timeout):
            raise TimeoutError(f"Rate limit exceeded (max {self.max_calls} calls per {self.time_window}s)")
        
        yield
    
    def get_current_rate(self) -> float:
        """Get current call rate per second."""
        current_time = time.time()
        
        with self.lock:
            # Remove old calls
            while self.calls and current_time - self.calls[0] > self.time_window:
                self.calls.popleft()
            
            return len(self.calls) / self.time_window

class Timeout:
    """Timeout manager for operations."""
    
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time: Optional[float] = None
    
    def start(self) -> 'Timeout':
        """Start the timeout."""
        self.start_time = time.time()
        return self
    
    def check(self) -> None:
        """Check if timeout has been exceeded."""
        if self.start_time is None:
            raise RuntimeError("Timeout not started")
        
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutError(f"Operation timed out after {elapsed:.2f}s (limit: {self.timeout_seconds}s)")
    
    @property
    def remaining(self) -> float:
        """Get remaining time before timeout."""
        if self.start_time is None:
            return self.timeout_seconds
        
        elapsed = time.time() - self.start_time
        return max(0.0, self.timeout_seconds - elapsed)
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        
        return time.time() - self.start_time
    
    @contextmanager
    def limit(self):
        """Context manager for timeout."""
        self.start()
        try:
            yield self
        finally:
            self.check()

def timeout_after(seconds: float):
    """Decorator to add timeout to function calls."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timeout(seconds).limit():
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def async_timeout_after(seconds: float):
    """Decorator to add timeout to async function calls."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Async operation timed out after {seconds}s")
        
        return wrapper
    return decorator

# Global profiler instance
_global_profiler = PerformanceProfiler()

def profile_operation(operation: str, operation_id: str = None):
    """Decorator to profile function execution using global profiler."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _global_profiler.time_operation(operation, operation_id):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler