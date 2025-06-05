"""
Type definitions for adaptive speculative decoding.

This module contains all type hints, data classes, and type aliases
used throughout the system for better type safety and documentation.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import time
from abc import ABC, abstractmethod

# Type aliases for common types
TokenId = int
TokenIds = List[TokenId]
LogProb = float
LogProbs = List[LogProb]
Timestamp = float
ModelName = str
StageId = str

class TaskDomain(Enum):
    """Task domain classification."""
    MATHEMATICAL = "mathematical"
    TECHNICAL = "technical"
    FACTUAL = "factual"
    CREATIVE = "creative"
    REASONING = "reasoning"
    ANALYTICAL = "analytical"
    LINGUISTIC = "linguistic"
    CONVERSATIONAL = "conversational"

class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
    RESEARCH = "research"

class CognitiveLoad(Enum):
    """Cognitive processing requirements."""
    RECALL = "recall"
    COMPREHENSION = "comprehension"
    APPLICATION = "application"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"

@dataclass(frozen=True)
class TaskCharacteristics:
    """Comprehensive task characteristics."""
    domain: TaskDomain
    complexity: TaskComplexity
    cognitive_load: CognitiveLoad
    
    # Detailed attributes
    requires_computation: bool = False
    requires_creativity: bool = False
    requires_factual_knowledge: bool = False
    requires_reasoning: bool = False
    requires_code_generation: bool = False
    
    # Quantitative measures
    estimated_tokens: int = 0
    estimated_steps: int = 1
    domain_expertise_level: float = 0.0
    
    # Contextual factors
    has_constraints: bool = False
    requires_examples: bool = False
    benefits_from_iteration: bool = False

@dataclass
class ModelOutput:
    """Output from a model stage."""
    text: str
    token_ids: TokenIds
    logprobs: Optional[LogProbs] = None
    
    # Generation metadata
    generation_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Quality indicators
    confidence_score: Optional[float] = None
    perplexity: Optional[float] = None
    
    # Timing information
    timestamp: Timestamp = field(default_factory=time.time)
    
    @property
    def total_tokens(self) -> int:
        """Total number of tokens (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens
    
    @property
    def tokens_per_second(self) -> float:
        """Generation speed in tokens per second."""
        if self.generation_time <= 0:
            return 0.0
        return self.completion_tokens / self.generation_time

@dataclass
class PredictionResult:
    """Result from quality prediction."""
    predicted_quality: float
    confidence: float
    uncertainty: Optional[float] = None
    
    # Model-specific predictions (for ensemble)
    individual_predictions: Optional[Dict[str, float]] = None
    
    # Feature information
    feature_importance: Optional[Dict[str, float]] = None
    
    # Metadata
    prediction_time: float = 0.0
    model_version: Optional[str] = None
    timestamp: Timestamp = field(default_factory=time.time)

@dataclass
class QualityMetrics:
    """Comprehensive quality evaluation metrics."""
    # Primary metrics
    bleu_score: Optional[float] = None
    rouge1_score: Optional[float] = None
    rouge2_score: Optional[float] = None
    rougeL_score: Optional[float] = None
    bertscore_f1: Optional[float] = None
    
    # Secondary metrics
    meteor_score: Optional[float] = None
    semantic_coherence: Optional[float] = None
    repetition_score: Optional[float] = None
    vocabulary_diversity: Optional[float] = None
    
    # Task-specific metrics
    task_specific_scores: Dict[str, float] = field(default_factory=dict)
    
    # Aggregate scores
    aggregate_score: Optional[float] = None
    weighted_score: Optional[float] = None
    
    # Evaluation metadata
    reference_available: bool = False
    evaluation_time: float = 0.0
    timestamp: Timestamp = field(default_factory=time.time)
    
    def get_primary_score(self) -> float:
        """Get the primary quality score."""
        if self.aggregate_score is not None:
            return self.aggregate_score
        elif self.bertscore_f1 is not None:
            return self.bertscore_f1
        elif self.bleu_score is not None:
            return self.bleu_score
        else:
            return 0.0

@dataclass
class SystemMetrics:
    """System performance and resource metrics."""
    # Performance metrics
    request_count: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput_qps: float = 0.0
    
    # Quality metrics
    avg_quality: float = 0.0
    quality_std: float = 0.0
    min_quality: float = 0.0
    max_quality: float = 0.0
    
    # Cost metrics
    avg_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    
    # Resource metrics
    gpu_utilization: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    queue_lengths: Dict[str, int] = field(default_factory=dict)
    
    # Model usage distribution
    stage_usage_count: Dict[str, int] = field(default_factory=dict)
    stage_usage_percentage: Dict[str, float] = field(default_factory=dict)
    
    # Error tracking
    error_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Timing information
    measurement_window: float = 0.0
    timestamp: Timestamp = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 0.0
        return self.successful_requests / total

@dataclass
class OptimizationDecision:
    """Decision made by the optimization algorithm."""
    selected_stage: StageId
    expected_quality: float
    expected_cost: float
    expected_latency: float
    
    # Decision reasoning
    stage_scores: Dict[StageId, float]
    decision_factors: Dict[str, Any]
    
    # Context
    lambda_param: float
    system_load: Dict[str, float]
    task_characteristics: TaskCharacteristics
    
    # Metadata
    decision_time: float = 0.0
    algorithm_version: str = "unknown"
    timestamp: Timestamp = field(default_factory=time.time)

@dataclass
class RequestContext:
    """Context information for a request."""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0
    
    # Task information
    task_characteristics: Optional[TaskCharacteristics] = None
    expected_domain: Optional[TaskDomain] = None
    
    # Quality requirements
    min_quality_threshold: float = 0.8
    max_latency_ms: Optional[float] = None
    cost_budget: Optional[float] = None
    
    # Client information
    client_id: Optional[str] = None
    priority: int = 0
    
    # Metadata
    created_at: Timestamp = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate request context after initialization."""
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

@dataclass
class ProcessingResult:
    """Complete result of processing a request."""
    request_context: RequestContext
    model_output: ModelOutput
    optimization_decision: OptimizationDecision
    prediction_result: PredictionResult
    quality_metrics: QualityMetrics
    
    # Processing metadata
    total_processing_time: float = 0.0
    stages_used: List[StageId] = field(default_factory=list)
    early_stopped: bool = False
    
    # Cost tracking
    total_cost: float = 0.0
    cost_breakdown: Dict[StageId, float] = field(default_factory=dict)
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    
    # Timing breakdown
    time_breakdown: Dict[str, float] = field(default_factory=dict)
    
    timestamp: Timestamp = field(default_factory=time.time)

# Generic type variables
T = TypeVar('T')
ConfigType = TypeVar('ConfigType')
ResultType = TypeVar('ResultType')

# Protocol for configuration classes
class Configurable(Protocol):
    """Protocol for classes that can be configured."""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the object with given configuration."""
        ...
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        ...

# Protocol for metrics collection
class MetricsCollector(Protocol):
    """Protocol for metrics collection."""
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        ...
    
    def reset_metrics(self) -> None:
        """Reset metrics collection."""
        ...

# Type aliases for common function signatures
QualityPredictionFunction = callable[[str, int], PredictionResult]
OptimizationFunction = callable[[RequestContext, Dict[str, Any]], OptimizationDecision]
QualityEvaluationFunction = callable[[ModelOutput, Optional[str]], QualityMetrics]