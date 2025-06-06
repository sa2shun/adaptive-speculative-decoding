# Optimal Stopping for Hierarchical Language Model Inference

This repository contains the implementation for our paper on theoretically-grounded adaptive speculative decoding.

## Quick Start

```bash
# Install dependencies
pip install -r requirements_minimal.txt

# Run core experiments (takes ~2 hours on 8 GPUs)
python run_core_experiments.py --experiments all

# Generate paper figure
python create_paper_figure.py results/paper_experiments/
```

## Theoretical Guarantees

Our method provides:
- **Regret Bound**: R(T) ≤ O(√nT log T)
- **Sample Complexity**: O(1/ε² log(n/δ)) for (ε,δ)-optimal policy
- **Computational**: O(n) per decision

## Minimal Implementation

```python
from src.minimal_adaptive_decoder import MinimalAdaptiveDecoder

# Initialize with theoretical optimal thresholds
decoder = MinimalAdaptiveDecoder("configs/qwen3_models.yaml")

# Adaptive inference
result = decoder.decode("Explain quantum computing")
print(f"Used: Qwen3-{result.selected_stage}, Speedup: {10/result.cost:.1f}x")
```

## Reproducing Paper Results

### 1. Regret Analysis (Figure 1a)
```bash
python run_core_experiments.py --experiments regret
```
Validates theoretical O(√T log T) bound empirically.

### 2. Quality-Cost Tradeoff (Figure 1b)
```bash
python run_core_experiments.py --experiments tradeoff
```
Shows Pareto-optimal performance across λ values.

### 3. Statistical Tests (Table 1)
```bash
python run_core_experiments.py --experiments statistical
```
Produces LaTeX table with p-values and effect sizes.

## Model Configuration

Using Qwen3 family for theoretical consistency:
- **Qwen3-7B**: Base model (cost=1.0)
- **Qwen3-14B**: 2x cost, +10% quality
- **Qwen3-32B**: 4.5x cost, +15% quality  
- **Qwen3-72B**: 10x cost, +20% quality

All models use INT8 quantization for memory efficiency.

## Baselines

We compare against theoretically motivated baselines:
- **Oracle**: Upper bound with perfect knowledge
- **Fixed-k**: Traditional single-model approach
- **Random**: Lower bound baseline
- **Threshold**: Simple confidence-based stopping

## Key Files

- `src/theory/`: Theoretical framework and proofs
- `src/minimal_adaptive_decoder.py`: Core implementation
- `src/statistical_evaluation.py`: Rigorous hypothesis testing
- `run_core_experiments.py`: Reproduce all paper results

## Citation

```bibtex
@article{optimal-stopping-llm-2024,
  title={Optimal Stopping for Hierarchical Language Model Inference},
  author={...},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- 8 GPUs (for full Qwen3-72B)
- ~300GB disk space for models

## License

Apache 2.0