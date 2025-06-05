"""
System-wide configuration for adaptive speculative decoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import os

from .base import BaseConfig, ResourceConfig, ConfigurationError

@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for logging system."""
    
    # Log levels
    root_level: str = "INFO"
    module_levels: Dict[str, str] = field(default_factory=lambda: {
        'adaptive_sd': 'INFO',
        'vllm': 'WARNING',
        'transformers': 'WARNING',
        'torch': 'WARNING'
    })
    
    # Log formats
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    
    # Output configuration
    log_dir: str = "logs"
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # Component-specific logging
    log_requests: bool = True
    log_model_loading: bool = True
    log_optimization_decisions: bool = True
    log_quality_predictions: bool = False  # Can be verbose
    
    # Performance logging
    log_latencies: bool = True
    log_gpu_metrics: bool = True
    log_memory_usage: bool = True
    
    def validate(self) -> None:
        """Validate logging configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        if self.root_level not in valid_levels:
            raise ConfigurationError(f"Invalid root_level: {self.root_level}")
        
        for module, level in self.module_levels.items():
            if level not in valid_levels:
                raise ConfigurationError(f"Invalid level for {module}: {level}")
        
        if self.max_file_size_mb <= 0:
            raise ConfigurationError("max_file_size_mb must be positive")

@dataclass
class SecurityConfig(BaseConfig):
    """Configuration for security features."""
    
    # API security
    enable_api_key_auth: bool = False
    api_key_header: str = "X-API-Key"
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 1000
    
    # Input validation
    max_input_length: int = 8192
    allowed_file_types: List[str] = field(default_factory=lambda: ['.txt', '.json'])
    sanitize_inputs: bool = True
    
    # Model security
    trust_remote_code: bool = False
    allow_model_downloads: bool = False
    verify_model_checksums: bool = True
    
    # Network security
    allowed_hosts: List[str] = field(default_factory=lambda: ['localhost', '127.0.0.1'])
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ['*'])
    
    def validate(self) -> None:
        """Validate security configuration."""
        if self.max_requests_per_minute <= 0:
            raise ConfigurationError("max_requests_per_minute must be positive")
        
        if self.max_input_length <= 0:
            raise ConfigurationError("max_input_length must be positive")

@dataclass
class MonitoringConfig(BaseConfig):
    """Configuration for monitoring and observability."""
    
    # Metrics collection
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    metrics_retention_days: int = 30
    
    # Health checks
    health_check_timeout: float = 5.0
    health_check_endpoints: List[str] = field(default_factory=lambda: [
        '/health',
        '/health/ready',
        '/health/live'
    ])
    
    # Performance monitoring
    track_request_latencies: bool = True
    track_model_utilization: bool = True
    track_memory_usage: bool = True
    track_error_rates: bool = True
    
    # Alerting (if integrated with external systems)
    enable_alerting: bool = False
    alert_webhook_url: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'error_rate': 0.05,
        'avg_latency_ms': 1000.0,
        'gpu_utilization': 0.95,
        'memory_usage': 0.9
    })
    
    # Distributed tracing
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    trace_sampling_rate: float = 0.1
    
    def validate(self) -> None:
        """Validate monitoring configuration."""
        if not 1 <= self.prometheus_port <= 65535:
            raise ConfigurationError("prometheus_port must be between 1 and 65535")
        
        if self.health_check_timeout <= 0:
            raise ConfigurationError("health_check_timeout must be positive")
        
        if not 0.0 <= self.trace_sampling_rate <= 1.0:
            raise ConfigurationError("trace_sampling_rate must be between 0 and 1")

@dataclass
class EnvironmentConfig(BaseConfig):
    """Configuration for environment-specific settings."""
    
    # Environment type
    environment: str = "development"  # development, staging, production
    debug_mode: bool = True
    
    # CUDA configuration
    cuda_visible_devices: Optional[str] = None
    cuda_memory_fraction: float = 0.9
    
    # Networking
    bind_address: str = "0.0.0.0"
    external_url: Optional[str] = None
    
    # File paths (with environment variable expansion)
    data_root: str = "/raid/$USER/adaptive-sd-data"
    model_root: str = "/raid/$USER/adaptive-sd-models"
    cache_root: str = "/raid/$USER/adaptive-sd-cache"
    temp_dir: str = "/tmp/adaptive-sd"
    
    # Resource limits
    max_memory_gb: Optional[float] = None
    max_disk_gb: Optional[float] = None
    
    def validate(self) -> None:
        """Validate environment configuration."""
        valid_environments = ['development', 'staging', 'production']
        if self.environment not in valid_environments:
            raise ConfigurationError(f"Invalid environment: {self.environment}")
        
        if not 0.0 < self.cuda_memory_fraction <= 1.0:
            raise ConfigurationError("cuda_memory_fraction must be between 0 and 1")
    
    def expand_paths(self) -> None:
        """Expand environment variables in paths."""
        self.data_root = os.path.expandvars(self.data_root)
        self.model_root = os.path.expandvars(self.model_root)
        self.cache_root = os.path.expandvars(self.cache_root)
        self.temp_dir = os.path.expandvars(self.temp_dir)

@dataclass
class SystemConfig(BaseConfig):
    """Complete system configuration."""
    
    # Sub-configurations
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Global system settings
    system_name: str = "adaptive-speculative-decoding"
    version: str = "2.0.0"
    
    # Worker configuration
    num_workers: int = 1
    worker_timeout: float = 300.0
    
    # Shutdown configuration
    graceful_shutdown_timeout: float = 30.0
    force_shutdown_timeout: float = 60.0
    
    def validate(self) -> None:
        """Validate system configuration."""
        # Validate sub-configurations
        self.resources.validate()
        self.logging.validate()
        self.security.validate()
        self.monitoring.validate()
        self.environment.validate()
        
        if self.num_workers <= 0:
            raise ConfigurationError("num_workers must be positive")
        
        if self.worker_timeout <= 0:
            raise ConfigurationError("worker_timeout must be positive")
        
        # Cross-validation between configurations
        total_gpus = sum(self.resources.tensor_parallel_size.values())
        if total_gpus > self.resources.gpu_count:
            raise ConfigurationError(
                f"Total tensor parallel size ({total_gpus}) "
                f"exceeds available GPUs ({self.resources.gpu_count})"
            )
    
    def setup_environment(self) -> None:
        """Setup environment based on configuration."""
        # Expand environment variables
        self.environment.expand_paths()
        
        # Create necessary directories
        directories = [
            self.environment.data_root,
            self.environment.model_root,
            self.environment.cache_root,
            self.environment.temp_dir,
            self.logging.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Set CUDA environment variables
        if self.environment.cuda_visible_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.environment.cuda_visible_devices
        
        # Set memory fraction
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:512'
    
    def get_effective_config(self) -> Dict[str, any]:
        """Get the effective configuration with all overrides applied."""
        config_dict = self.to_dict()
        
        # Apply environment-specific overrides
        if self.environment.environment == "production":
            config_dict['logging']['root_level'] = 'WARNING'
            config_dict['environment']['debug_mode'] = False
            config_dict['security']['enable_api_key_auth'] = True
        elif self.environment.environment == "development":
            config_dict['logging']['root_level'] = 'DEBUG'
            config_dict['environment']['debug_mode'] = True
            
        return config_dict