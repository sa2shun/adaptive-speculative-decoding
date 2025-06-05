"""
Interfaces and abstract base classes for adaptive speculative decoding.

This module defines the contracts that all components must implement,
ensuring consistency and enabling dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, AsyncIterator
import asyncio

from .types import (
    RequestContext, ProcessingResult, PredictionResult, QualityMetrics,
    ModelOutput, OptimizationDecision, SystemMetrics, TaskCharacteristics
)
from .exceptions import AdaptiveDecodingError

class QualityPredictor(ABC):
    """Abstract interface for quality prediction models."""
    
    @abstractmethod
    def predict_quality(self, prompt: str, stage_id: str) -> PredictionResult:
        """
        Predict the quality of output for a given prompt and stage.
        
        Args:
            prompt: Input prompt text
            stage_id: Identifier of the model stage
            
        Returns:
            PredictionResult containing quality prediction and metadata
            
        Raises:
            PredictionError: If prediction fails
        """
        pass
    
    @abstractmethod
    def predict_quality_batch(self, prompts: List[str], stage_ids: List[str]) -> List[PredictionResult]:
        """
        Predict quality for multiple prompt-stage pairs.
        
        Args:
            prompts: List of input prompts
            stage_ids: List of stage identifiers
            
        Returns:
            List of PredictionResult objects
        """
        pass
    
    @abstractmethod
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the quality predictor on provided data.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Training metrics and metadata
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the predictor has been trained."""
        pass

class ModelStage(ABC):
    """Abstract interface for individual model stages."""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100, 
                temperature: float = 0.0, top_p: float = 1.0) -> ModelOutput:
        """
        Generate text using this model stage.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            ModelOutput containing generated text and metadata
        """
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, max_tokens: int = 100,
                           temperature: float = 0.0, top_p: float = 1.0) -> ModelOutput:
        """Async version of generate method."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, max_tokens: int = 100,
                       temperature: float = 0.0, top_p: float = 1.0) -> AsyncIterator[str]:
        """
        Generate text with streaming output.
        
        Yields:
            Incremental text chunks as they are generated
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
    
    @property
    @abstractmethod
    def stage_id(self) -> str:
        """Get the stage identifier."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this stage."""
        pass

class OptimizationStrategy(ABC):
    """Abstract interface for optimization algorithms."""
    
    @abstractmethod
    def select_stage(self, context: RequestContext, 
                    available_stages: List[str],
                    predictions: Dict[str, PredictionResult],
                    system_state: Dict[str, Any]) -> OptimizationDecision:
        """
        Select the optimal stage for processing a request.
        
        Args:
            context: Request context with prompt and requirements
            available_stages: List of available stage identifiers
            predictions: Quality predictions for each stage
            system_state: Current system state and load information
            
        Returns:
            OptimizationDecision with selected stage and reasoning
        """
        pass
    
    @abstractmethod
    def update_parameters(self, feedback: Dict[str, Any]) -> None:
        """
        Update optimization parameters based on feedback.
        
        Args:
            feedback: Performance feedback and metrics
        """
        pass
    
    @abstractmethod
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current optimization parameters."""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Get the strategy name."""
        pass

class QualityEvaluator(ABC):
    """Abstract interface for quality evaluation."""
    
    @abstractmethod
    def evaluate(self, output: ModelOutput, 
                reference: Optional[str] = None,
                task_type: Optional[str] = None) -> QualityMetrics:
        """
        Evaluate the quality of model output.
        
        Args:
            output: Model output to evaluate
            reference: Optional reference text for comparison
            task_type: Optional task type for specialized evaluation
            
        Returns:
            QualityMetrics containing evaluation results
        """
        pass
    
    @abstractmethod
    def evaluate_batch(self, outputs: List[ModelOutput],
                      references: Optional[List[str]] = None,
                      task_types: Optional[List[str]] = None) -> List[QualityMetrics]:
        """Evaluate multiple outputs in batch."""
        pass
    
    @abstractmethod
    def compare_outputs(self, output1: ModelOutput, output2: ModelOutput,
                       reference: Optional[str] = None) -> Dict[str, float]:
        """
        Compare two outputs and return comparison metrics.
        
        Returns:
            Dictionary with comparison scores
        """
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported evaluation metrics."""
        pass

class CacheManager(ABC):
    """Abstract interface for cache management."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all items from cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

class MetricsCollector(ABC):
    """Abstract interface for metrics collection."""
    
    @abstractmethod
    def record_request(self, context: RequestContext, result: ProcessingResult) -> None:
        """Record metrics for a completed request."""
        pass
    
    @abstractmethod
    def record_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Record error metrics."""
        pass
    
    @abstractmethod
    def get_metrics(self, time_window: Optional[float] = None) -> SystemMetrics:
        """Get current system metrics."""
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        pass
    
    @abstractmethod
    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        pass

class TaskClassifier(ABC):
    """Abstract interface for task classification."""
    
    @abstractmethod
    def classify(self, prompt: str) -> TaskCharacteristics:
        """
        Classify a prompt into task characteristics.
        
        Args:
            prompt: Input prompt to classify
            
        Returns:
            TaskCharacteristics with classification results
        """
        pass
    
    @abstractmethod
    def classify_batch(self, prompts: List[str]) -> List[TaskCharacteristics]:
        """Classify multiple prompts in batch."""
        pass
    
    @abstractmethod
    def get_model_recommendation(self, characteristics: TaskCharacteristics) -> Dict[str, float]:
        """
        Get model recommendations based on task characteristics.
        
        Returns:
            Dictionary mapping stage IDs to recommendation scores
        """
        pass

class LoadBalancer(ABC):
    """Abstract interface for load balancing."""
    
    @abstractmethod
    def select_instance(self, stage_id: str, request_context: RequestContext) -> str:
        """
        Select the best instance for processing a request.
        
        Args:
            stage_id: Identifier of the model stage
            request_context: Context of the request
            
        Returns:
            Instance identifier
        """
        pass
    
    @abstractmethod
    def update_load(self, instance_id: str, load_delta: float) -> None:
        """Update load information for an instance."""
        pass
    
    @abstractmethod
    def get_load_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get load statistics for all instances."""
        pass

class Pipeline(ABC):
    """Abstract interface for the main processing pipeline."""
    
    @abstractmethod
    def process_request(self, context: RequestContext) -> ProcessingResult:
        """
        Process a single request through the adaptive pipeline.
        
        Args:
            context: Request context with prompt and parameters
            
        Returns:
            ProcessingResult with output and metadata
        """
        pass
    
    @abstractmethod
    async def process_request_async(self, context: RequestContext) -> ProcessingResult:
        """Async version of process_request."""
        pass
    
    @abstractmethod
    def process_stream(self, context: RequestContext) -> AsyncIterator[str]:
        """Process request with streaming output."""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        pass
    
    @abstractmethod
    def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shutdown the pipeline."""
        pass

# Factory interfaces for dependency injection
class ComponentFactory(ABC):
    """Abstract factory for creating pipeline components."""
    
    @abstractmethod
    def create_quality_predictor(self, config: Dict[str, Any]) -> QualityPredictor:
        """Create a quality predictor instance."""
        pass
    
    @abstractmethod
    def create_model_stage(self, config: Dict[str, Any]) -> ModelStage:
        """Create a model stage instance."""
        pass
    
    @abstractmethod
    def create_optimization_strategy(self, config: Dict[str, Any]) -> OptimizationStrategy:
        """Create an optimization strategy instance."""
        pass
    
    @abstractmethod
    def create_quality_evaluator(self, config: Dict[str, Any]) -> QualityEvaluator:
        """Create a quality evaluator instance."""
        pass
    
    @abstractmethod
    def create_task_classifier(self, config: Dict[str, Any]) -> TaskClassifier:
        """Create a task classifier instance."""
        pass

# Mixin classes for common functionality
class ConfigurableMixin:
    """Mixin for components that can be configured."""
    
    def __init__(self):
        self._config = {}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component."""
        self._config.update(config)
        self._validate_config()
        self._apply_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()
    
    def _validate_config(self) -> None:
        """Validate configuration. Override in subclasses."""
        pass
    
    def _apply_config(self) -> None:
        """Apply configuration. Override in subclasses."""
        pass

class HealthCheckMixin:
    """Mixin for components that support health checks."""
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'status': 'healthy' if self._is_healthy() else 'unhealthy',
            'timestamp': __import__('time').time(),
            'details': self._get_health_details()
        }
    
    def _is_healthy(self) -> bool:
        """Check if component is healthy. Override in subclasses."""
        return True
    
    def _get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information. Override in subclasses."""
        return {}

# Type aliases for common interface combinations
QualityPredictorFactory = callable[[Dict[str, Any]], QualityPredictor]
ModelStageFactory = callable[[Dict[str, Any]], ModelStage]
OptimizationStrategyFactory = callable[[Dict[str, Any]], OptimizationStrategy]