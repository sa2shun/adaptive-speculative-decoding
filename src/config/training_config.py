"""
Training and evaluation configuration for adaptive speculative decoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from enum import Enum

from .base import BaseConfig, ConfigurationError

class PredictorType(Enum):
    """Types of quality predictors."""
    MLP = "mlp"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"

class DatasetType(Enum):
    """Types of evaluation datasets."""
    MMLU = "mmlu"
    HUMANEVAL = "humaneval"
    GSM8K = "gsm8k"
    TRUTHFULQA = "truthfulqa"
    HELLASWAG = "hellaswag"
    ARC = "arc"
    CUSTOM = "custom"

@dataclass
class DataGenerationConfig(BaseConfig):
    """Configuration for training data generation."""
    
    # Data generation parameters
    num_samples: int = 100_000
    sample_distribution: Dict[str, float] = field(default_factory=lambda: {
        'simple': 0.6,
        'moderate': 0.3,
        'complex': 0.1
    })
    
    # Query generation
    max_prompt_length: int = 512
    min_prompt_length: int = 10
    include_code_queries: bool = True
    include_math_queries: bool = True
    include_reasoning_queries: bool = True
    
    # Quality target simulation
    quality_noise_std: float = 0.02
    latency_noise_std: float = 10.0
    
    # Domain distribution
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        'factual': 0.25,
        'reasoning': 0.20,
        'technical': 0.20,
        'creative': 0.15,
        'mathematical': 0.10,
        'conversational': 0.10
    })
    
    # Output paths
    output_dir: str = "/raid/$USER/adaptive-sd-training-data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    def validate(self) -> None:
        """Validate data generation configuration."""
        if self.num_samples <= 0:
            raise ConfigurationError("num_samples must be positive")
        
        # Validate sample distribution sums to 1
        if abs(sum(self.sample_distribution.values()) - 1.0) > 1e-6:
            raise ConfigurationError("sample_distribution must sum to 1.0")
        
        # Validate domain weights sum to 1
        if abs(sum(self.domain_weights.values()) - 1.0) > 1e-6:
            raise ConfigurationError("domain_weights must sum to 1.0")
        
        # Validate split ratios
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ConfigurationError("train/val/test splits must sum to 1.0")

@dataclass
class PredictorTrainingConfig(BaseConfig):
    """Configuration for quality predictor training."""
    
    # Model configuration
    predictor_type: PredictorType = PredictorType.ENSEMBLE
    random_seed: int = 42
    
    # Training parameters
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Cross-validation
    cv_folds: int = 5
    validation_split: float = 0.2
    
    # Ensemble configuration
    ensemble_models: List[str] = field(default_factory=lambda: [
        'random_forest',
        'gradient_boosting', 
        'lightgbm',
        'neural_network',
        'ridge'
    ])
    
    # Neural network specific
    hidden_layers: Tuple[int, ...] = (256, 128, 64, 32)
    activation: str = "relu"
    dropout_rate: float = 0.2
    batch_normalization: bool = True
    
    # Tree-based model parameters
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 3
    
    # Feature engineering
    feature_scaling: bool = True
    feature_selection: bool = True
    max_features: Optional[int] = None
    
    # Output configuration
    model_output_dir: str = "/raid/$USER/adaptive-sd-models/predictors"
    save_best_only: bool = True
    save_feature_importance: bool = True
    
    def validate(self) -> None:
        """Validate predictor training configuration."""
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        
        if self.num_epochs <= 0:
            raise ConfigurationError("num_epochs must be positive")
        
        if not 0.0 < self.learning_rate <= 1.0:
            raise ConfigurationError("learning_rate must be between 0 and 1")
        
        if not 0.0 <= self.validation_split < 1.0:
            raise ConfigurationError("validation_split must be between 0 and 1")
        
        if self.cv_folds < 2:
            raise ConfigurationError("cv_folds must be at least 2")

@dataclass
class EvaluationDatasetConfig(BaseConfig):
    """Configuration for a single evaluation dataset."""
    
    name: str
    dataset_type: DatasetType
    path: Optional[str] = None
    subset: Optional[str] = None
    split: str = "test"
    max_samples: Optional[int] = None
    
    # Task-specific configuration
    task_category: str = "general"
    expected_output_length: int = 100
    has_ground_truth: bool = True
    
    # Evaluation parameters
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        'bleu', 'rouge1', 'rougeL', 'bertscore'
    ])
    
    def validate(self) -> None:
        """Validate evaluation dataset configuration."""
        if not self.name:
            raise ConfigurationError("Dataset name cannot be empty")
        
        if self.max_samples is not None and self.max_samples <= 0:
            raise ConfigurationError("max_samples must be positive")

@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation."""
    
    # Datasets to evaluate
    datasets: List[EvaluationDatasetConfig] = field(default_factory=lambda: [
        EvaluationDatasetConfig(
            name="mmlu",
            dataset_type=DatasetType.MMLU,
            task_category="factual",
            max_samples=1000
        ),
        EvaluationDatasetConfig(
            name="humaneval", 
            dataset_type=DatasetType.HUMANEVAL,
            task_category="technical",
            max_samples=500
        ),
        EvaluationDatasetConfig(
            name="gsm8k",
            dataset_type=DatasetType.GSM8K, 
            task_category="mathematical",
            max_samples=500
        )
    ])
    
    # Evaluation parameters
    batch_size: int = 32
    max_concurrent_requests: int = 10
    timeout_per_request: float = 60.0
    
    # Lambda sweep configuration  
    lambda_values: List[float] = field(default_factory=lambda: [
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0
    ])
    
    # Baseline comparisons
    include_single_model_baselines: bool = True
    include_static_pipeline: bool = True
    
    # Statistical analysis
    num_bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    significance_level: float = 0.01
    
    # Output configuration
    results_dir: str = "/raid/$USER/adaptive-sd-results"
    save_individual_results: bool = True
    save_aggregated_results: bool = True
    generate_visualizations: bool = True
    
    def validate(self) -> None:
        """Validate evaluation configuration."""
        if not self.datasets:
            raise ConfigurationError("At least one dataset must be configured")
        
        for i, dataset in enumerate(self.datasets):
            try:
                dataset.validate()
            except ConfigurationError as e:
                raise ConfigurationError(f"Invalid dataset {i}: {e}")
        
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        
        if not 0.0 < self.confidence_level < 1.0:
            raise ConfigurationError("confidence_level must be between 0 and 1")
        
        if not 0.0 < self.significance_level < 1.0:
            raise ConfigurationError("significance_level must be between 0 and 1")

@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration for experimental runs."""
    
    # Experiment identification
    experiment_name: str = "default_experiment"
    experiment_description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Reproducibility
    random_seed: int = 42
    deterministic_mode: bool = True
    
    # Resource allocation
    max_gpu_memory_fraction: float = 0.9
    parallel_experiments: int = 1
    
    # Logging and monitoring
    log_interval: int = 100
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000
    
    # Output management
    output_dir: str = "/raid/$USER/adaptive-sd-experiments"
    overwrite_existing: bool = False
    compress_outputs: bool = True
    
    def validate(self) -> None:
        """Validate experiment configuration."""
        if not self.experiment_name:
            raise ConfigurationError("experiment_name cannot be empty")
        
        if not 0.0 < self.max_gpu_memory_fraction <= 1.0:
            raise ConfigurationError("max_gpu_memory_fraction must be between 0 and 1")
        
        if self.parallel_experiments <= 0:
            raise ConfigurationError("parallel_experiments must be positive")

@dataclass
class TrainingConfig(BaseConfig):
    """Complete training configuration."""
    
    # Sub-configurations
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    predictor_training: PredictorTrainingConfig = field(default_factory=PredictorTrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Global training parameters
    use_mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Distributed training
    enable_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    def validate(self) -> None:
        """Validate training configuration."""
        # Validate sub-configurations
        self.data_generation.validate()
        self.predictor_training.validate()
        self.evaluation.validate()
        self.experiment.validate()
        
        if self.gradient_clipping <= 0:
            raise ConfigurationError("gradient_clipping must be positive")
        
        if self.world_size <= 0:
            raise ConfigurationError("world_size must be positive")
        
        if not 0 <= self.rank < self.world_size:
            raise ConfigurationError("rank must be between 0 and world_size-1")