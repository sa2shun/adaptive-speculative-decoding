"""
Tests for configuration management system.
"""

import pytest
import yaml
import json
from pathlib import Path
from typing import Dict, Any

from src.config.base import BaseConfig, ConfigManager, ConfigurationError
from src.config.model_config import ModelConfig, StageConfig
from src.config.serving_config import ServingConfig
from src.config.training_config import TrainingConfig
from src.config.system_config import SystemConfig

class TestBaseConfig:
    """Test the base configuration functionality."""
    
    def test_base_config_creation(self):
        """Test basic configuration creation."""
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "test"
            value: int = 42
        
        config = TestConfig()
        assert config.name == "test"
        assert config.value == 42
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "default"
            value: int = 0
        
        data = {"name": "test", "value": 100}
        config = TestConfig.from_dict(data)
        
        assert config.name == "test"
        assert config.value == 100
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "test"
            value: int = 42
        
        config = TestConfig()
        data = config.to_dict()
        
        assert data == {"name": "test", "value": 42}
    
    def test_config_validation_error(self):
        """Test configuration validation errors."""
        
        @dataclass
        class TestConfig(BaseConfig):
            value: int = 0
            
            def validate(self):
                if self.value < 0:
                    raise ConfigurationError("Value must be non-negative")
        
        # Valid configuration should not raise
        config = TestConfig(value=10)
        config.validate()  # Should not raise
        
        # Invalid configuration should raise
        config = TestConfig(value=-1)
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_config_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file."""
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "default"
            value: int = 0
        
        # Create test YAML file
        yaml_content = """
        name: "test_config"
        value: 123
        """
        yaml_file = temp_dir / "test.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Load configuration
        config = TestConfig.from_yaml(yaml_file)
        assert config.name == "test_config"
        assert config.value == 123
    
    def test_config_save_yaml(self, temp_dir):
        """Test saving configuration to YAML file."""
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "test"
            value: int = 42
        
        config = TestConfig()
        yaml_file = temp_dir / "output.yaml"
        
        # Save configuration
        config.save_yaml(yaml_file)
        
        # Verify file was created and contains correct data
        assert yaml_file.exists()
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data == {"name": "test", "value": 42}

class TestModelConfig:
    """Test model configuration functionality."""
    
    def test_stage_config_validation(self):
        """Test stage configuration validation."""
        
        # Valid stage configuration
        stage = StageConfig(
            model_name="test-model",
            model_path="/path/to/model",
            size_label="13B",
            tensor_parallel_size=1,
            base_cost=1.0
        )
        stage.validate()  # Should not raise
        
        # Invalid stage configuration (empty model name)
        with pytest.raises(ConfigurationError):
            stage = StageConfig(
                model_name="",
                model_path="/path/to/model",
                size_label="13B"
            )
            stage.validate()
        
        # Invalid stage configuration (negative cost)
        with pytest.raises(ConfigurationError):
            stage = StageConfig(
                model_name="test-model",
                model_path="/path/to/model",
                size_label="13B",
                base_cost=-1.0
            )
            stage.validate()
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        
        # Valid model configuration
        stage1 = StageConfig(
            model_name="model1",
            model_path="/path1",
            size_label="13B",
            tensor_parallel_size=1,
            gpu_ids=[0]
        )
        stage2 = StageConfig(
            model_name="model2", 
            model_path="/path2",
            size_label="34B",
            tensor_parallel_size=2,
            gpu_ids=[1, 2]
        )
        
        config = ModelConfig(stages=[stage1, stage2])
        config.validate()  # Should not raise
        
        # Invalid: overlapping GPU IDs
        stage2_invalid = StageConfig(
            model_name="model2",
            model_path="/path2", 
            size_label="34B",
            tensor_parallel_size=2,
            gpu_ids=[0, 1]  # Overlaps with stage1
        )
        
        with pytest.raises(ConfigurationError):
            config = ModelConfig(stages=[stage1, stage2_invalid])
            config.validate()
    
    def test_model_config_stage_lookup(self):
        """Test stage lookup functionality."""
        
        stage1 = StageConfig(
            model_name="model1",
            model_path="/path1",
            size_label="13B"
        )
        stage2 = StageConfig(
            model_name="model2",
            model_path="/path2",
            size_label="34B"
        )
        
        config = ModelConfig(stages=[stage1, stage2])
        
        # Test lookup by label
        found_stage = config.get_stage_by_label("13B")
        assert found_stage is not None
        assert found_stage.model_name == "model1"
        
        # Test lookup by index
        found_stage = config.get_stage_by_index(1)
        assert found_stage is not None
        assert found_stage.size_label == "34B"
        
        # Test non-existent lookup
        assert config.get_stage_by_label("70B") is None
        assert config.get_stage_by_index(10) is None

class TestServingConfig:
    """Test serving configuration functionality."""
    
    def test_optimization_config_validation(self):
        """Test optimization configuration validation."""
        
        from src.config.serving_config import OptimizationConfig
        
        # Valid configuration
        config = OptimizationConfig(
            lambda_param=1.0,
            target_latency_ms=200.0,
            max_error_rate=0.01
        )
        config.validate()  # Should not raise
        
        # Invalid lambda parameter
        with pytest.raises(ConfigurationError):
            config = OptimizationConfig(lambda_param=-1.0)
            config.validate()
        
        # Invalid cost multiplier range
        with pytest.raises(ConfigurationError):
            config = OptimizationConfig(
                max_cost_multiplier=1.0,
                min_cost_multiplier=2.0  # max < min
            )
            config.validate()
    
    def test_server_config_validation(self):
        """Test server configuration validation."""
        
        from src.config.serving_config import ServerConfig
        
        # Valid configuration
        config = ServerConfig(
            host="localhost",
            port=8000,
            max_concurrent_requests=100
        )
        config.validate()  # Should not raise
        
        # Invalid port
        with pytest.raises(ConfigurationError):
            config = ServerConfig(port=0)
            config.validate()
        
        with pytest.raises(ConfigurationError):
            config = ServerConfig(port=70000)
            config.validate()

class TestTrainingConfig:
    """Test training configuration functionality."""
    
    def test_data_generation_config_validation(self):
        """Test data generation configuration validation."""
        
        from src.config.training_config import DataGenerationConfig
        
        # Valid configuration
        config = DataGenerationConfig(
            num_samples=1000,
            sample_distribution={'simple': 0.6, 'moderate': 0.3, 'complex': 0.1}
        )
        config.validate()  # Should not raise
        
        # Invalid sample distribution (doesn't sum to 1)
        with pytest.raises(ConfigurationError):
            config = DataGenerationConfig(
                sample_distribution={'simple': 0.5, 'moderate': 0.3, 'complex': 0.3}
            )
            config.validate()
    
    def test_predictor_training_config_validation(self):
        """Test predictor training configuration validation."""
        
        from src.config.training_config import PredictorTrainingConfig
        
        # Valid configuration
        config = PredictorTrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            cv_folds=5
        )
        config.validate()  # Should not raise
        
        # Invalid batch size
        with pytest.raises(ConfigurationError):
            config = PredictorTrainingConfig(batch_size=0)
            config.validate()
        
        # Invalid cross-validation folds
        with pytest.raises(ConfigurationError):
            config = PredictorTrainingConfig(cv_folds=1)
            config.validate()

class TestSystemConfig:
    """Test system configuration functionality."""
    
    def test_logging_config_validation(self):
        """Test logging configuration validation."""
        
        from src.config.system_config import LoggingConfig
        
        # Valid configuration
        config = LoggingConfig(
            root_level="INFO",
            module_levels={"test": "DEBUG"}
        )
        config.validate()  # Should not raise
        
        # Invalid log level
        with pytest.raises(ConfigurationError):
            config = LoggingConfig(root_level="INVALID")
            config.validate()
    
    def test_security_config_validation(self):
        """Test security configuration validation."""
        
        from src.config.system_config import SecurityConfig
        
        # Valid configuration
        config = SecurityConfig(
            max_requests_per_minute=1000,
            max_input_length=8192
        )
        config.validate()  # Should not raise
        
        # Invalid request rate limit
        with pytest.raises(ConfigurationError):
            config = SecurityConfig(max_requests_per_minute=0)
            config.validate()

class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_config_manager_load(self, temp_dir):
        """Test configuration manager loading."""
        
        # Create test configuration file
        config_data = {
            "system_name": "test",
            "version": "1.0.0"
        }
        
        config_file = temp_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create config manager with test directory
        manager = ConfigManager(config_dir=temp_dir)
        
        # Load configuration (using BaseConfig for simplicity)
        from dataclasses import dataclass
        
        @dataclass
        class TestConfig(BaseConfig):
            system_name: str = "default"
            version: str = "0.0.0"
        
        config = manager.load_config("test", TestConfig)
        
        assert config.system_name == "test"
        assert config.version == "1.0.0"
    
    def test_config_manager_fallback_to_default(self, temp_dir):
        """Test configuration manager fallback to defaults."""
        
        from dataclasses import dataclass
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "default"
            value: int = 42
        
        # Create config manager with empty directory
        manager = ConfigManager(config_dir=temp_dir)
        
        # Load non-existent configuration - should use defaults
        config = manager.load_config("nonexistent", TestConfig, fallback_to_env=False)
        
        assert config.name == "default"
        assert config.value == 42
    
    def test_config_manager_caching(self, temp_dir):
        """Test configuration manager caching."""
        
        from dataclasses import dataclass
        
        @dataclass
        class TestConfig(BaseConfig):
            name: str = "test"
        
        manager = ConfigManager(config_dir=temp_dir)
        
        # Load configuration twice
        config1 = manager.load_config("test", TestConfig, fallback_to_env=False)
        config2 = manager.get_config("test")
        
        # Should return the same cached instance
        assert config2 is not None
        assert config1.name == config2.name

@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_system_config_loading(self, temp_dir):
        """Test loading complete system configuration."""
        
        # Create comprehensive configuration
        config_data = {
            "system": {
                "system_name": "adaptive-sd-test",
                "version": "2.0.0",
                "num_workers": 2
            },
            "models": {
                "stages": [
                    {
                        "model_name": "test-13B",
                        "model_path": "/test/path",
                        "size_label": "13B",
                        "tensor_parallel_size": 1,
                        "gpu_ids": [0]
                    }
                ]
            },
            "serving": {
                "optimization": {
                    "lambda_param": 2.0
                },
                "server": {
                    "port": 9000
                }
            }
        }
        
        config_file = temp_dir / "unified.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(config_dir=temp_dir)
        
        # Load different configuration sections
        system_config = manager.load_config("unified", SystemConfig)
        
        # Verify configuration was loaded correctly
        assert system_config.system_name == "adaptive-sd-test"
        assert system_config.version == "2.0.0"
        assert system_config.num_workers == 2
    
    def test_config_validation_cascade(self):
        """Test that validation errors cascade properly."""
        
        # Create invalid configuration that should fail validation
        invalid_stage = StageConfig(
            model_name="",  # Invalid: empty name
            model_path="/path",
            size_label="13B"
        )
        
        invalid_model_config = ModelConfig(stages=[invalid_stage])
        
        # Should raise validation error
        with pytest.raises(ConfigurationError) as exc_info:
            invalid_model_config.validate()
        
        # Error message should mention the specific validation failure
        assert "model_name cannot be empty" in str(exc_info.value)