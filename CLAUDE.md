# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL RESEARCH QUALITY REQUIREMENTS ⚠️

**This research is a once‐in‐a‐lifetime, critically important endeavor. Under no circumstances should you compromise its quality by, for example, making the model lightweight or reducing the dataset size. Please carry out the research rigorously and document it thoroughly.**

### Mandatory Quality Standards:
- **NO COMPROMISES** on model size, dataset scale, or experimental rigor
- Use FULL-PRECISION Qwen3 models (7B, 14B, 32B, 72B) without quantization
- Maintain LARGE-SCALE datasets (100K+ training samples, 1000+ evaluation samples per task)
- Conduct COMPREHENSIVE experiments across all complexity levels and λ values
- Document ALL experimental details, hyperparameters, and results with research-grade precision
- Ensure reproducibility with fixed seeds and detailed environment specifications
- **REAL MODEL EXECUTION ONLY** - no simulation or mock components

## Project Overview

This is the adaptive-speculative-decoding repository, implementing a multi-stage Draft-Verify pipeline with input-dependent depth optimization for Large Language Model (LLM) inference. The system uses **Qwen3 7B→14B→32B→72B model hierarchy** with dynamic stopping based on input difficulty and real latency-based cost modeling.

## Model Hierarchy Specification

### Fixed 4-Stage Qwen3 Architecture:
1. **Stage 0**: Qwen/Qwen3-7B-Instruct (7B parameters)
2. **Stage 1**: Qwen/Qwen3-14B-Instruct (14B parameters) 
3. **Stage 2**: Qwen/Qwen3-32B-Instruct (32B parameters)
4. **Stage 3**: Qwen/Qwen3-72B-Instruct (72B parameters)

### Technical Requirements:
- **No quantization**: Full FP16/BF16 precision for research accuracy
- **Tensor Parallelism**: Distributed across multiple GPUs as needed
- **Real Inference**: All experiments use actual model execution, no simulation
- **Cost Modeling**: Latency-based costs measured from actual inference runs

## Key Commands

### Environment Setup
```bash
bash scripts/setup_env.sh
conda activate adaptive-sd
pip install -r requirements.txt
```

### Model Download
```bash
# Download all Qwen3 models to /raid/$USER/adaptive-sd-models/
python scripts/download_qwen3_models.py --storage-path /raid/$USER/adaptive-sd-models/
```

### Training Data Generation
```bash
# Generate 100K training samples for quality predictor
python scripts/generate_training_data.py --num-samples 100000 --output-path /raid/$USER/adaptive-sd-training-data/
```

### Quality Predictor Training  
```bash
# Train quality predictor with real model data
python scripts/train_predictor.py --config configs/training.yaml
```

### Full Experimental Pipeline
```bash
# Run complete experiments (takes 24-48 hours on 8xA100)
bash experiments/run_complete_experiments.sh

# Or run comprehensive evaluation notebook
jupyter notebook experiments/comprehensive_evaluation.ipynb
```

### Evaluation
```bash
# Evaluate specific configuration
python experiments/evaluate_pipeline.py --datasets mmlu,humaneval,gsm8k,truthfulqa --lambda 1.0

# Run baseline comparisons  
python experiments/run_baseline_comparison.py
```

### Testing
```bash
pytest tests/ -v
mypy src/ --config-file mypy.ini
black src/ tests/ experiments/
flake8 src/ tests/
```

## Architecture

### Core Components
1. **Stage Model** (`src/models/stage.py`): Wrapper for each Qwen3 model in the hierarchy
2. **Quality Predictor** (`src/models/predictor.py`): MLP predicting acceptance probability using real features
3. **DP Solver** (`src/algorithms/dp_solver.py`): Dynamic programming for optimal stopping with latency-based costs
4. **Pipeline** (`src/serving/pipeline.py`): Orchestrates multi-stage inference with real-time decision making
5. **Cost Profiler** (`src/utils/cost_profiler.py`): Measures actual inference latency for cost modeling

### Key Design Decisions
- **vLLM** for efficient inference with tensor parallelism
- **No quantization** to maintain research-grade precision
- **Dynamic stopping** based on measured latency costs + quality loss minimization
- **Real cost modeling** from empirical latency measurements, not theoretical estimates
- **Bayesian risk adjustment** for prediction uncertainty

### Configuration Files
- `configs/qwen3_models.yaml`: Qwen3 model specifications and GPU allocation
- `configs/training.yaml`: Quality predictor training hyperparameters
- `configs/evaluation.yaml`: Comprehensive evaluation settings
- `configs/cost_profiling.yaml`: Latency measurement configuration

## Implementation Details

### Lambda Parameter
The λ (lambda) parameter controls quality-speed tradeoff:
- λ ∈ [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]: Comprehensive experimental sweep
- λ < 1: Prioritize speed (more early stops)
- λ = 1: Balanced cost-quality optimization
- λ > 5: Prioritize quality (fewer early stops)

### GPU Memory Allocation
**Full-precision requirements (no quantization):**
- 7B model: ~14GB GPU memory (1x A100)
- 14B model: ~28GB GPU memory (1x A100 or 2x A100 with TP=2)
- 32B model: ~64GB GPU memory (2x A100 with TP=2)
- 72B model: ~144GB GPU memory (4x A100 with TP=4)

**Total cluster requirement: 8x A100 (80GB each) for concurrent inference**

### Cost Modeling
1. **Latency Profiling**: Measure actual inference time for each model on representative samples
2. **Cost Vector**: [c₀, c₁, c₂, c₃] where cᵢ is measured latency for stage i
3. **Dynamic Updates**: Cost measurements updated during evaluation
4. **Hardware-Specific**: Costs measured on actual deployment hardware

### Quality Predictor Features
1. **Input complexity**: Token entropy, sequence length
2. **Linguistic features**: Syntactic depth, semantic difficulty  
3. **Context information**: Previous stage outputs, confidence scores
4. **Stage-specific**: Model-specific hidden state features

## Research-Grade Experimental Protocol

### Required Experimental Scope:
1. **Model Scale**: Full Qwen3 7B→14B→32B→72B hierarchy (NO compromises)
2. **Training Data**: Exactly 100,000 diverse, high-quality samples
3. **Evaluation Data**: 2,000+ samples per dataset (MMLU, HumanEval, GSM8K, TruthfulQA)
4. **Lambda Values**: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] (6 values comprehensive sweep)
5. **Baseline Comparisons**: Single-model inference for each stage with identical conditions
6. **Statistical Significance**: 5 independent runs with different seeds (1-5)
7. **Ablation Studies**: Quality predictor, cost modeling, stopping criteria

### Documentation Requirements:
- Complete experimental logs with GPU utilization, memory usage, timestamps
- Detailed hyperparameter configurations for reproducibility
- Performance metrics with 95% confidence intervals
- Real resource costs (GPU hours, energy consumption)
- Version-controlled experiment tracking

### Quality Metrics:
- **Primary**: Inference speedup vs single-model baselines (measured wall-clock time)
- **Quality**: Task-specific metrics (accuracy, pass@1, BLEU, ROUGE) with significance tests
- **Efficiency**: Tokens per second, cost per quality point
- **Analysis**: Stage utilization distribution, decision boundary visualization

### Storage and Data Management
```bash
# Storage hierarchy on 30TB /raid disk
/raid/$USER/adaptive-sd-models/          # Qwen3 models (~500GB total)
/raid/$USER/adaptive-sd-training-data/   # 100K training samples (~50GB)
/raid/$USER/adaptive-sd-eval-data/       # Evaluation datasets (~10GB)
/raid/$USER/adaptive-sd-results/         # Experimental results (~100GB)
/raid/$USER/adaptive-sd-logs/            # Complete experimental logs (~20GB)
```

### Computational Resources
- **Model Storage**: ~500GB for all Qwen3 models
- **Training Data**: ~50GB for 100K diverse samples
- **Evaluation**: 48-72 hours on 8x A100 cluster
- **Memory**: 640GB GPU memory total (8x 80GB A100)

## Experiment Execution

### Phase 1: Setup and Validation (4 hours)
```bash
bash experiments/setup_complete_environment.sh
python experiments/validate_model_access.py  # Verify all models load
python experiments/calibrate_cost_model.py   # Measure baseline latencies
```

### Phase 2: Training Data Generation (8 hours)
```bash
python scripts/generate_training_data.py --num-samples 100000 --real-inference
python scripts/train_quality_predictor.py --full-scale
```

### Phase 3: Comprehensive Evaluation (36 hours)
```bash
jupyter notebook experiments/comprehensive_evaluation.ipynb
# OR
bash experiments/run_complete_experiments.sh
```

### Phase 4: Analysis and Validation (4 hours)
```bash
python experiments/analyze_results.py --statistical-tests
python experiments/generate_paper_figures.py
```

## Quality Assurance

### Validation Checks:
1. **No simulation**: Verify all results from real model execution
2. **Reproducibility**: Multiple seeds produce consistent results within confidence intervals
3. **Baseline validity**: Single-model baselines match published benchmarks
4. **Cost accuracy**: Measured latencies reflect actual deployment conditions
5. **Statistical power**: Sample sizes sufficient for significance testing

### Common Issues and Solutions:

1. **GPU OOM**: Use /raid storage, ensure no quantization, optimize tensor parallelism
2. **Slow evaluation**: Batch processing, asynchronous inference where possible
3. **Inconsistent costs**: Re-calibrate cost model, check hardware consistency
4. **Poor predictor**: Increase training data quality, verify feature engineering

**CRITICAL**: Any deviation from this protocol must be documented and justified. This research allows no compromises on scale, precision, or rigor.