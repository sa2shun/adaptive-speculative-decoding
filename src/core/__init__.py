"""
Core modules for Adaptive Speculative Decoding.

This package contains the fundamental building blocks and interfaces
for the adaptive speculative decoding system.
"""

from .interfaces import (
    QualityPredictor,
    ModelStage,
    OptimizationStrategy,
    QualityEvaluator
)

from .exceptions import (
    AdaptiveDecodingError,
    ModelLoadError,
    PredictionError,
    ConfigurationError
)

from .types import (
    PredictionResult,
    QualityMetrics,
    ModelOutput,
    SystemMetrics
)

__all__ = [
    # Interfaces
    'QualityPredictor',
    'ModelStage', 
    'OptimizationStrategy',
    'QualityEvaluator',
    
    # Exceptions
    'AdaptiveDecodingError',
    'ModelLoadError',
    'PredictionError',
    'ConfigurationError',
    
    # Types
    'PredictionResult',
    'QualityMetrics',
    'ModelOutput',
    'SystemMetrics'
]