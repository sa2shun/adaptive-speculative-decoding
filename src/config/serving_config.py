"""
Serving configuration for adaptive speculative decoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum

from .base import BaseConfig, ConfigurationError

class OptimizationStrategy(Enum):
    """Optimization strategy for model selection."""
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    ENSEMBLE = "ensemble"

class QualityMetric(Enum):
    """Quality evaluation metrics."""
    PROBABILITY = "probability"
    BLEU = "bleu"
    ROUGE = "rouge"
    BERTSCORE = "bertscore"
    MULTI_METRIC = "multi_metric"

@dataclass
class OptimizationConfig(BaseConfig):
    """Configuration for the optimization algorithm."""
    
    # Core algorithm
    strategy: OptimizationStrategy = OptimizationStrategy.GREEDY
    lambda_param: float = 1.0
    
    # Dynamic cost adjustment
    enable_dynamic_costs: bool = True
    cost_adjustment_interval: float = 30.0  # seconds
    max_cost_multiplier: float = 3.0
    min_cost_multiplier: float = 0.5
    
    # Quality prediction
    quality_predictor_type: str = "ensemble"  # "simple", "ensemble", "neural"
    prediction_confidence_threshold: float = 0.8
    enable_uncertainty_quantification: bool = True
    
    # Performance targets
    target_latency_ms: float = 200.0
    max_error_rate: float = 0.01
    min_quality_score: float = 0.85
    target_throughput_qps: float = 10.0
    
    # Load balancing
    enable_load_balancing: bool = True
    queue_length_threshold: int = 10
    gpu_utilization_threshold: float = 0.85
    
    # 70B utilization enhancement
    enable_70b_enhancement: bool = True
    target_70b_utilization: float = 0.4
    complexity_detection_threshold: float = 0.6
    quality_critical_patterns: List[str] = field(default_factory=lambda: [
        r'\b(legal|medical|financial|safety)',
        r'\b(critical|important|essential)',
        r'\b(research|academic|scholarly)'
    ])
    
    def validate(self) -> None:
        """Validate optimization configuration."""
        if self.lambda_param <= 0:
            raise ConfigurationError("lambda_param must be positive")
        
        if not 0.0 < self.prediction_confidence_threshold <= 1.0:
            raise ConfigurationError("prediction_confidence_threshold must be between 0 and 1")
        
        if self.cost_adjustment_interval <= 0:
            raise ConfigurationError("cost_adjustment_interval must be positive")
        
        if self.max_cost_multiplier <= self.min_cost_multiplier:
            raise ConfigurationError("max_cost_multiplier must be greater than min_cost_multiplier")

@dataclass
class QualityConfig(BaseConfig):
    """Configuration for quality evaluation."""
    
    # Primary quality metrics
    primary_metric: QualityMetric = QualityMetric.MULTI_METRIC
    metrics_to_compute: List[QualityMetric] = field(default_factory=lambda: [
        QualityMetric.BLEU,
        QualityMetric.ROUGE, 
        QualityMetric.BERTSCORE
    ])
    
    # Metric weights for aggregation
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'bleu': 0.25,
        'rouge1': 0.15,
        'rougeL': 0.15,
        'bertscore_f1': 0.25,
        'semantic_coherence': 0.10,
        'repetition_score': 0.05,
        'general_quality': 0.05
    })
    
    # Quality evaluation parameters
    reference_data_path: Optional[str] = None
    enable_task_specific_metrics: bool = True
    statistical_significance_level: float = 0.01
    
    # Performance vs quality trade-off
    min_quality_for_early_stop: float = 0.8
    quality_degradation_threshold: float = 0.05
    
    def validate(self) -> None:
        """Validate quality configuration."""
        if not 0.0 < self.statistical_significance_level < 1.0:
            raise ConfigurationError("statistical_significance_level must be between 0 and 1")
        
        # Validate metric weights sum to 1
        if abs(sum(self.metric_weights.values()) - 1.0) > 1e-6:
            raise ConfigurationError("Metric weights must sum to 1.0")

@dataclass 
class ServerConfig(BaseConfig):
    """Configuration for the serving server."""
    
    # Network configuration
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Request handling
    max_concurrent_requests: int = 100
    request_timeout_seconds: float = 300.0
    max_request_size_mb: float = 10.0
    
    # Response configuration
    response_streaming: bool = True
    chunk_size: int = 1024
    
    # Health checks
    health_check_interval: float = 30.0
    enable_metrics_endpoint: bool = True
    metrics_port: int = 8001
    
    # Logging
    log_level: str = "INFO"
    access_log: bool = True
    error_log_path: Optional[str] = "logs/error.log"
    access_log_path: Optional[str] = "logs/access.log"
    
    def validate(self) -> None:
        """Validate server configuration."""
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        
        if self.max_concurrent_requests <= 0:
            raise ConfigurationError("max_concurrent_requests must be positive")
        
        if self.request_timeout_seconds <= 0:
            raise ConfigurationError("request_timeout_seconds must be positive")

@dataclass
class CacheConfig(BaseConfig):
    """Configuration for caching system."""
    
    # KV-cache configuration
    enable_kv_cache: bool = True
    kv_cache_size_gb: float = 32.0
    cache_eviction_policy: str = "lru"  # "lru", "fifo", "lfu"
    
    # Response caching
    enable_response_cache: bool = True
    response_cache_size_mb: float = 1024.0
    response_cache_ttl_seconds: int = 3600
    
    # Model caching
    preload_all_models: bool = True
    model_cache_cleanup_interval: float = 600.0
    
    def validate(self) -> None:
        """Validate cache configuration."""
        if self.kv_cache_size_gb <= 0:
            raise ConfigurationError("kv_cache_size_gb must be positive")
        
        if self.response_cache_ttl_seconds <= 0:
            raise ConfigurationError("response_cache_ttl_seconds must be positive")

@dataclass
class ServingConfig(BaseConfig):
    """Complete serving configuration."""
    
    # Sub-configurations
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Pipeline configuration
    enable_pipeline_parallelism: bool = True
    max_pipeline_depth: int = 3
    pipeline_timeout_seconds: float = 60.0
    
    # Monitoring and debugging
    enable_detailed_metrics: bool = True
    enable_request_tracing: bool = False
    debug_mode: bool = False
    profiling_enabled: bool = False
    
    # Production features
    enable_graceful_shutdown: bool = True
    shutdown_timeout_seconds: float = 30.0
    enable_auto_scaling: bool = False
    
    def validate(self) -> None:
        """Validate serving configuration."""
        # Validate sub-configurations
        self.optimization.validate()
        self.quality.validate()
        self.server.validate()
        self.cache.validate()
        
        if self.max_pipeline_depth <= 0:
            raise ConfigurationError("max_pipeline_depth must be positive")
        
        if self.pipeline_timeout_seconds <= 0:
            raise ConfigurationError("pipeline_timeout_seconds must be positive")