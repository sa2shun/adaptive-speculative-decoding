"""
Test suite for Adaptive Speculative Decoding.

This package contains comprehensive tests covering all aspects of the system:
- Unit tests for individual components
- Integration tests for system interactions
- Performance benchmarks
- End-to-end testing
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from pathlib import Path
import tempfile
import shutil

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_CONFIG_DIR = Path(__file__).parent / "configs"
TEST_MODELS_DIR = Path(__file__).parent / "models"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_CONFIG_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "system": {
            "system_name": "adaptive-sd-test",
            "version": "2.0.0-test",
            "num_workers": 1
        },
        "models": {
            "stages": [
                {
                    "model_name": "test-model-13B",
                    "model_path": str(TEST_MODELS_DIR / "test-13B"),
                    "size_label": "13B",
                    "tensor_parallel_size": 1,
                    "base_latency_ms": 100.0,
                    "base_cost": 1.0
                }
            ]
        },
        "serving": {
            "optimization": {
                "lambda_param": 1.0,
                "enable_dynamic_costs": False
            },
            "server": {
                "host": "127.0.0.1",
                "port": 8080
            }
        }
    }

# Test markers
pytestmark = [
    pytest.mark.asyncio,  # Mark all tests as async by default
]