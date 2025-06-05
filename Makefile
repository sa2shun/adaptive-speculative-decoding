# Makefile for Adaptive Speculative Decoding
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev install-research install-prod
.PHONY: test test-unit test-integration test-benchmark test-coverage
.PHONY: lint format type-check quality-check pre-commit
.PHONY: clean clean-build clean-pyc clean-test
.PHONY: docs docs-build docs-serve
.PHONY: build publish
.PHONY: setup-env download-models
.PHONY: run-server run-experiments run-evaluation
.PHONY: docker-build docker-run docker-stop

# Default target
help:
	@echo "Adaptive Speculative Decoding - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install package with basic dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  install-research Install with research dependencies"
	@echo "  install-prod     Install with production dependencies"
	@echo "  setup-env        Setup development environment"
	@echo "  download-models  Download required models"
	@echo ""
	@echo "Development Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-benchmark   Run performance benchmarks"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  quality-check    Run all quality checks"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs-build       Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Application Commands:"
	@echo "  run-server       Start the adaptive SD server"
	@echo "  run-experiments  Run experimental evaluation"
	@echo "  run-evaluation   Run model evaluation"
	@echo ""
	@echo "Build Commands:"
	@echo "  build            Build distribution packages"
	@echo "  publish          Publish to PyPI"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-stop      Stop Docker container"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  clean            Clean all build artifacts"
	@echo "  clean-build      Clean build artifacts"
	@echo "  clean-pyc        Clean Python cache files"
	@echo "  clean-test       Clean test artifacts"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-research:
	pip install -e ".[research]"

install-prod:
	pip install -e ".[production]"

install-all:
	pip install -e ".[all]"

# Environment setup
setup-env:
	@echo "Setting up development environment..."
	bash scripts/setup_env.sh
	pip install -e ".[dev]"
	pre-commit install
	@echo "Development environment setup complete!"

download-models:
	@echo "Downloading required models..."
	python scripts/download_models.py --all
	@echo "Model download complete!"

# Testing targets
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-benchmark:
	pytest tests/benchmarks/ -v --tb=short --benchmark-only

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

test-parallel:
	pytest tests/ -n auto -v

test-fast:
	pytest tests/ -x -v --tb=short

# Code quality targets
lint:
	flake8 src tests scripts
	pylint src

format:
	black src tests scripts examples
	isort src tests scripts examples

format-check:
	black --check src tests scripts examples
	isort --check-only src tests scripts examples

type-check:
	mypy src --config-file pyproject.toml

quality-check: format-check lint type-check
	@echo "All quality checks passed!"

pre-commit:
	pre-commit run --all-files

# Documentation targets
docs-build:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

docs-clean:
	cd docs && make clean

# Application targets
run-server:
	python -m src.serving.server --config configs/serving.yaml

run-server-dev:
	python -m src.serving.server --config configs/serving.yaml --debug

run-experiments:
	python run_comprehensive_evaluation.py

run-evaluation:
	python experiments/evaluate_pipeline.py --datasets mmlu humaneval gsm8k

run-training:
	python scripts/train_predictor.py --config configs/training.yaml

# Build targets
clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

clean-pyc:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -name "*.pyo" -delete

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

clean: clean-build clean-pyc clean-test
	rm -rf logs/
	rm -rf .tox/
	rm -rf .cache/

build: clean-build
	python -m build

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*

# Docker targets
DOCKER_IMAGE = adaptive-sd
DOCKER_TAG = latest
DOCKER_CONTAINER = adaptive-sd-container

docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -d --name $(DOCKER_CONTAINER) \
		--gpus all \
		-p 8000:8000 \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-stop:
	docker stop $(DOCKER_CONTAINER) || true
	docker rm $(DOCKER_CONTAINER) || true

docker-logs:
	docker logs -f $(DOCKER_CONTAINER)

docker-shell:
	docker exec -it $(DOCKER_CONTAINER) bash

# Research and experimentation targets
experiment-baseline:
	python run_baseline_comparison.py --output results/baseline_$(shell date +%Y%m%d_%H%M%S).json

experiment-ablation:
	python experiments/ablation/run_ablation_study.py

experiment-full:
	bash experiments/run_full_evaluation.sh

benchmark-performance:
	python -m pytest tests/benchmarks/ --benchmark-only --benchmark-save=performance

benchmark-memory:
	python -m memory_profiler scripts/benchmark_memory.py

# Model management targets
models-list:
	python scripts/download_models.py --list

models-clean:
	rm -rf /raid/$$USER/adaptive-sd-models/*
	@echo "Model cache cleared"

models-validate:
	python scripts/validate_models.py

# Configuration management
config-validate:
	python -c "from src.config.base import ConfigManager; ConfigManager('configs').load_config('unified', dict)"

config-generate:
	python scripts/generate_config.py --output configs/generated.yaml

# Monitoring and profiling
profile-server:
	python -m cProfile -o profiles/server.prof -m src.serving.server

profile-inference:
	python -m cProfile -o profiles/inference.prof scripts/profile_inference.py

monitor-gpu:
	watch -n 1 nvidia-smi

# Development utilities
format-imports:
	isort src tests scripts examples

check-security:
	bandit -r src/

check-dependencies:
	pip-audit

update-dependencies:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Release management
version-bump-patch:
	bump2version patch

version-bump-minor:
	bump2version minor

version-bump-major:
	bump2version major

release-prepare:
	@echo "Preparing release..."
	$(MAKE) quality-check
	$(MAKE) test
	$(MAKE) docs-build
	@echo "Release preparation complete!"

# Environment variables
export PYTHONPATH := $(PYTHONPATH):$(PWD)/src
export ADAPTIVE_SD_CONFIG_DIR := $(PWD)/configs
export ADAPTIVE_SD_LOG_LEVEL := INFO