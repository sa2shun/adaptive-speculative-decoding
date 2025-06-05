"""
Model configuration for adaptive speculative decoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path

from .base import BaseConfig, ConfigurationError

@dataclass
class StageConfig(BaseConfig):
    """Configuration for a single model stage."""
    
    # Model identification
    model_name: str
    model_path: str
    size_label: str  # "13B", "34B", "70B"
    
    # Model parameters
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    dtype: str = "auto"  # "auto", "float16", "bfloat16"
    
    # Performance parameters
    base_latency_ms: float = 100.0
    base_cost: float = 1.0
    base_capacity_qps: float = 10.0
    
    # Quality parameters
    quality_range_min: float = 0.8
    quality_range_max: float = 0.95
    
    # GPU assignment
    gpu_ids: List[int] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate stage configuration."""
        if not self.model_name:
            raise ConfigurationError("model_name cannot be empty")
        
        if not self.model_path:
            raise ConfigurationError("model_path cannot be empty")
        
        if self.tensor_parallel_size <= 0:
            raise ConfigurationError("tensor_parallel_size must be positive")
        
        if self.max_model_len <= 0:
            raise ConfigurationError("max_model_len must be positive")
        
        if not 0.0 < self.quality_range_min < self.quality_range_max <= 1.0:
            raise ConfigurationError("Invalid quality range")
        
        if self.base_cost <= 0:
            raise ConfigurationError("base_cost must be positive")

@dataclass
class ModelConfig(BaseConfig):
    """Configuration for all models in the pipeline."""
    
    # Model stages
    stages: List[StageConfig] = field(default_factory=lambda: [
        StageConfig(
            model_name="meta-llama/Llama-2-13b-chat-hf",
            model_path="/raid/$USER/adaptive-sd-models/llama-2-13b-chat",
            size_label="13B",
            tensor_parallel_size=1,
            base_latency_ms=120.0,
            base_cost=1.0,
            base_capacity_qps=50.0,
            quality_range_min=0.75,
            quality_range_max=0.90,
            gpu_ids=[0]
        ),
        StageConfig(
            model_name="codellama/CodeLlama-34b-Instruct-hf",
            model_path="/raid/$USER/adaptive-sd-models/codellama-34b-instruct",
            size_label="34B", 
            tensor_parallel_size=2,
            base_latency_ms=180.0,
            base_cost=1.3,
            base_capacity_qps=25.0,
            quality_range_min=0.85,
            quality_range_max=0.94,
            gpu_ids=[1, 2]
        ),
        StageConfig(
            model_name="meta-llama/Llama-2-70b-chat-hf",
            model_path="/raid/$USER/adaptive-sd-models/llama-2-70b-chat",
            size_label="70B",
            tensor_parallel_size=4,
            base_latency_ms=320.0,
            base_cost=1.8,
            base_capacity_qps=12.0,
            quality_range_min=0.90,
            quality_range_max=0.96,
            gpu_ids=[3, 4, 5, 6]
        )
    ])
    
    # Model loading parameters
    trust_remote_code: bool = False
    revision: str = "main"
    tokenizer_mode: str = "auto"  # "auto", "slow"
    skip_tokenizer_init: bool = False
    
    # Quantization settings
    quantization: Optional[str] = None  # None, "awq", "gptq", "squeezellm"
    load_format: str = "auto"  # "auto", "pt", "safetensors", "npy", "npz"
    
    # Memory optimization
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    
    # Model-specific overrides
    model_overrides: Dict[str, Dict[str, any]] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate model configuration."""
        if not self.stages:
            raise ConfigurationError("At least one stage must be configured")
        
        # Validate each stage
        for i, stage in enumerate(self.stages):
            try:
                stage.validate()
            except ConfigurationError as e:
                raise ConfigurationError(f"Invalid stage {i}: {e}")
        
        # Check for duplicate size labels
        size_labels = [stage.size_label for stage in self.stages]
        if len(size_labels) != len(set(size_labels)):
            raise ConfigurationError("Duplicate size labels found")
        
        # Validate GPU assignments don't overlap
        all_gpu_ids = []
        for stage in self.stages:
            all_gpu_ids.extend(stage.gpu_ids)
        
        if len(all_gpu_ids) != len(set(all_gpu_ids)):
            raise ConfigurationError("GPU IDs cannot be assigned to multiple stages")
        
        # Check tensor parallel size matches GPU assignment
        for stage in self.stages:
            if len(stage.gpu_ids) != stage.tensor_parallel_size:
                raise ConfigurationError(
                    f"Stage {stage.size_label}: GPU count ({len(stage.gpu_ids)}) "
                    f"doesn't match tensor_parallel_size ({stage.tensor_parallel_size})"
                )
    
    def get_stage_by_label(self, size_label: str) -> Optional[StageConfig]:
        """Get stage configuration by size label."""
        for stage in self.stages:
            if stage.size_label == size_label:
                return stage
        return None
    
    def get_stage_by_index(self, index: int) -> Optional[StageConfig]:
        """Get stage configuration by index."""
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return None
    
    @property
    def total_gpus_required(self) -> int:
        """Total number of GPUs required."""
        return sum(stage.tensor_parallel_size for stage in self.stages)
    
    @property
    def size_labels(self) -> List[str]:
        """Get all size labels."""
        return [stage.size_label for stage in self.stages]