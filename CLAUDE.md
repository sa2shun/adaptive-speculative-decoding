# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the adaptive-speculative-decoding repository, implementing a multi-stage Draft-Verify pipeline with input-dependent depth optimization for Large Language Model (LLM) inference. The system uses 8B→13B→34B→70B model hierarchy with dynamic stopping based on input difficulty.

## Key Commands

### Environment Setup
```bash
bash scripts/setup_env.sh
conda activate adaptive-sd
pip install -r requirements.txt
```

### Training
```bash
# Generate training data for quality predictor
python scripts/generate_training_data.py --num-samples 100000

# Train quality predictor
python scripts/train_predictor.py --config configs/training.yaml
```

### Serving
```bash
# Start the API server
python -m src.serving.server --config configs/serving.yaml

# Run demo
python examples/quick_demo.py
```

### Evaluation
```bash
# Run full evaluation
bash experiments/run_full_evaluation.sh

# Specific dataset evaluation
python experiments/evaluate_pipeline.py --datasets mmlu --lambda 1.0
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
1. **Stage Model** (`src/models/stage.py`): Wrapper for each LLM in the hierarchy
2. **Quality Predictor** (`src/models/predictor.py`): Lightweight MLP predicting acceptance probability
3. **DP Solver** (`src/algorithms/dp_solver.py`): Dynamic programming for optimal stopping
4. **Pipeline** (`src/serving/pipeline.py`): Orchestrates multi-stage inference with dynamic stopping
5. **KV-Cache Manager** (`src/serving/cache_manager.py`): Manages memory for interrupted inference

### Key Design Decisions
- **vLLM** for efficient inference with tensor parallelism
- **bitsandbytes** for 4-bit quantization to fit larger models
- **Dynamic stopping** based on expected cost + quality loss minimization
- **Risk adjustment** using Bayesian shrinkage for uncertain predictions

### Configuration
- `configs/models.yaml`: Model specifications and parallelism settings
- `configs/training.yaml`: Predictor training hyperparameters
- `configs/serving.yaml`: Server and pipeline runtime settings

## Important Implementation Details

### Lambda Parameter
The λ (lambda) parameter controls quality-speed tradeoff:
- λ < 1: Prioritize speed (more early stops)
- λ = 1: Balanced
- λ > 5: Prioritize quality (fewer early stops)

### GPU Memory Management
- 8B model: 1 GPU (TP=1)
- 13B model: 2 GPUs (TP=2)
- 34B model: 2 GPUs (TP=2)
- 70B model: 4 GPUs (TP=4)

### Quality Predictor Features
1. Input entropy (last 32 tokens)
2. Input/output length ratios
3. Average logprobs
4. Stage information

## Debugging Tips

### Check GPU allocation
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

### Monitor server logs
```bash
tail -f logs/server.log
```

### Profile inference
```python
from src.utils.debug import profile_inference
profile_inference()
```

## Common Issues

1. **OOM Error**: Reduce batch size or increase quantization
2. **Slow predictor**: Check feature extraction is < 0.3ms
3. **Poor quality**: Verify λ parameter and predictor accuracy