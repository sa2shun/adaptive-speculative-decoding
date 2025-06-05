"""
Adaptive Speculative Decoding

A multi-stage adaptive inference pipeline for Large Language Models with 
dynamic quality prediction and cost optimization.

This package provides:
- Multi-model stage pipeline (13B → 34B → 70B)
- Ensemble quality prediction with uncertainty quantification  
- Dynamic cost optimization and load balancing
- Comprehensive quality evaluation (BLEU, ROUGE, BERTScore)
- Production-ready serving infrastructure
- Research-grade experimentation framework

Example:
    Basic usage for inference:
    
    >>> from adaptive_sd import AdaptivePipeline
    >>> pipeline = AdaptivePipeline.from_config("configs/serving.yaml")
    >>> result = pipeline.generate("Explain quantum computing", max_tokens=100)
    >>> print(result.text)
    
    Advanced usage with custom parameters:
    
    >>> from adaptive_sd.config import ServingConfig
    >>> from adaptive_sd.serving import Pipeline
    >>> 
    >>> config = ServingConfig.from_yaml("configs/serving.yaml")
    >>> config.optimization.lambda_param = 2.0  # Favor quality
    >>> pipeline = Pipeline(config)
    >>> result = pipeline.process_request(RequestContext(
    ...     prompt="Design a distributed system",
    ...     max_tokens=500,
    ...     min_quality_threshold=0.9
    ... ))

Modules:
    config: Configuration management system
    core: Core interfaces and types
    models: Model implementations and quality prediction
    serving: Serving infrastructure and pipeline
    algorithms: Optimization algorithms
    evaluation: Quality evaluation and metrics
    utils: Utility functions and helpers

For detailed documentation, see: https://adaptive-sd.readthedocs.io/
"""

__version__ = "2.0.0"
__author__ = "Adaptive SD Research Team"
__email__ = "research@adaptive-sd.ai"
__license__ = "Apache-2.0"

# Core imports for convenient access
try:
    from .core.interfaces import Pipeline, QualityPredictor, ModelStage
    from .core.types import (
        RequestContext, 
        ProcessingResult, 
        TaskCharacteristics,
        QualityMetrics
    )
    from .core.exceptions import AdaptiveDecodingError

    # Configuration imports
    from .config.base import ConfigManager
    from .config.model_config import ModelConfig
    from .config.serving_config import ServingConfig  
    from .config.training_config import TrainingConfig
    from .config.system_config import SystemConfig
except ImportError as e:
    # Handle cases where modules might not be fully set up yet
    import warnings
    warnings.warn(f"Some modules not available during import: {e}", ImportWarning)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Core interfaces (if available)
    "Pipeline",
    "QualityPredictor", 
    "ModelStage",
    
    # Core types (if available)
    "RequestContext",
    "ProcessingResult",
    "TaskCharacteristics", 
    "QualityMetrics",
    
    # Exceptions (if available)
    "AdaptiveDecodingError",
    
    # Configuration (if available)
    "ConfigManager",
    "ModelConfig",
    "ServingConfig",
    "TrainingConfig", 
    "SystemConfig",
]

# Package metadata
__package_info__ = {
    "name": "adaptive-speculative-decoding",
    "version": __version__,
    "description": "Multi-stage adaptive inference pipeline for Large Language Models",
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": "https://github.com/sa2shun/adaptive-speculative-decoding",
    "documentation": "https://adaptive-sd.readthedocs.io/",
    "keywords": ["llm", "inference", "optimization", "adaptive", "speculative-decoding"],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
}

def get_version() -> str:
    """Get the package version."""
    return __version__

def get_package_info() -> dict:
    """Get complete package information."""
    return __package_info__.copy()

# Initialize logging when package is imported
import logging

# Create package logger
_logger = logging.getLogger(__name__)

# Add a null handler to prevent "No handler found" warnings
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

# Set up basic configuration if no handlers are configured for root logger
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

_logger.info(f"Adaptive Speculative Decoding v{__version__} initialized")

# Perform basic environment checks
def _check_environment():
    """Check if the environment is properly configured."""
    import sys
    import warnings
    
    # Check Python version
    if sys.version_info < (3, 10):
        warnings.warn(
            f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
            "Adaptive SD requires Python 3.10 or later.",
            RuntimeWarning
        )
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            _logger.info(f"CUDA available with {gpu_count} GPU(s)")
        else:
            _logger.warning("CUDA not available - CPU-only mode")
    except ImportError:
        _logger.warning("PyTorch not found - some features may not work")
    
    # Check for optional dependencies
    optional_deps = {
        "vllm": "High-performance inference",
        "bitsandbytes": "Model quantization", 
        "wandb": "Experiment tracking",
        "tensorboard": "Visualization"
    }
    
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            _logger.debug(f"Optional dependency '{dep}' not found ({desc})")

# Run environment checks
_check_environment()

# Cleanup
del _check_environment, _logger