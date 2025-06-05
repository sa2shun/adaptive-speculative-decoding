"""
Base configuration classes with validation and environment variable support.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass

@dataclass
class BaseConfig(ABC):
    """Base configuration class with common functionality."""
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        try:
            return cls(**data)
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration data: {e}")
    
    @classmethod
    def from_env(cls, prefix: str = "") -> 'BaseConfig':
        """Load configuration from environment variables."""
        env_data = {}
        
        for field_name in cls.__dataclass_fields__:
            env_key = f"{prefix}_{field_name}".upper() if prefix else field_name.upper()
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                # Type conversion based on field type
                field_type = cls.__dataclass_fields__[field_name].type
                try:
                    if field_type == bool:
                        env_data[field_name] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_type == int:
                        env_data[field_name] = int(env_value)
                    elif field_type == float:
                        env_data[field_name] = float(env_value)
                    elif field_type == List[str]:
                        env_data[field_name] = env_value.split(',')
                    else:
                        env_data[field_name] = env_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_key}={env_value}: {e}")
        
        return cls.from_dict(env_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, 'to_dict'):
                result[field_name] = field_value.to_dict()
            elif isinstance(field_value, (list, tuple)):
                result[field_name] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in field_value
                ]
            else:
                result[field_name] = field_value
        return result
    
    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def validate(self) -> None:
        """Validate configuration. Override in subclasses."""
        pass
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge with another configuration, other takes precedence."""
        if not isinstance(other, self.__class__):
            raise ConfigurationError(f"Cannot merge {type(self)} with {type(other)}")
        
        merged_data = self.to_dict()
        other_data = other.to_dict()
        
        def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_data = deep_merge(merged_data, other_data)
        return self.__class__.from_dict(merged_data)

@dataclass 
class ResourceConfig(BaseConfig):
    """Hardware resource configuration."""
    
    # GPU configuration
    gpu_count: int = 8
    gpu_memory_gb: float = 84.9
    tensor_parallel_size: Dict[str, int] = field(default_factory=lambda: {
        '13B': 1, '34B': 2, '70B': 4
    })
    
    # Memory configuration
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    swap_space_gb: int = 64
    
    # CPU configuration  
    cpu_cores: int = 64
    system_memory_gb: int = 1024
    
    # Storage configuration
    model_cache_path: str = "/raid/$USER/adaptive-sd-models"
    data_cache_path: str = "/raid/$USER/adaptive-sd-data"
    results_path: str = "/raid/$USER/adaptive-sd-results"
    
    def validate(self) -> None:
        """Validate resource configuration."""
        if self.gpu_count <= 0:
            raise ConfigurationError("gpu_count must be positive")
        
        if not 0.1 <= self.gpu_memory_utilization <= 1.0:
            raise ConfigurationError("gpu_memory_utilization must be between 0.1 and 1.0")
        
        # Validate tensor parallel configuration
        total_tp = sum(self.tensor_parallel_size.values())
        if total_tp > self.gpu_count:
            logger.warning(f"Total tensor parallel size ({total_tp}) exceeds GPU count ({self.gpu_count})")

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, BaseConfig] = {}
        self._env_prefix = "ADAPTIVE_SD"
    
    def load_config(self, config_name: str, config_class: type, 
                   fallback_to_env: bool = True) -> BaseConfig:
        """Load configuration with fallback hierarchy."""
        
        # Try loading from YAML file first
        yaml_path = self.config_dir / f"{config_name}.yaml"
        
        if yaml_path.exists():
            try:
                config = config_class.from_yaml(yaml_path)
                logger.info(f"Loaded {config_name} from {yaml_path}")
            except ConfigurationError as e:
                logger.error(f"Failed to load {config_name} from YAML: {e}")
                config = None
        else:
            config = None
        
        # Try environment variables as fallback
        if config is None and fallback_to_env:
            try:
                env_prefix = f"{self._env_prefix}_{config_name.upper()}"
                env_config = config_class.from_env(env_prefix)
                if env_config:
                    config = env_config
                    logger.info(f"Loaded {config_name} from environment variables")
            except ConfigurationError as e:
                logger.warning(f"Failed to load {config_name} from environment: {e}")
        
        # Use default configuration as final fallback
        if config is None:
            config = config_class()
            logger.info(f"Using default configuration for {config_name}")
        
        # Validate configuration
        config.validate()
        
        # Cache configuration
        self._configs[config_name] = config
        
        return config
    
    def get_config(self, config_name: str) -> Optional[BaseConfig]:
        """Get cached configuration."""
        return self._configs.get(config_name)
    
    def reload_config(self, config_name: str, config_class: type) -> BaseConfig:
        """Reload configuration from source."""
        if config_name in self._configs:
            del self._configs[config_name]
        return self.load_config(config_name, config_class)
    
    def save_all_configs(self, output_dir: Union[str, Path] = None) -> None:
        """Save all cached configurations to files."""
        if output_dir is None:
            output_dir = self.config_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for config_name, config in self._configs.items():
            output_path = output_dir / f"{config_name}.yaml"
            config.save_yaml(output_path)
            logger.info(f"Saved {config_name} configuration to {output_path}")

# Global configuration manager instance
config_manager = ConfigManager()