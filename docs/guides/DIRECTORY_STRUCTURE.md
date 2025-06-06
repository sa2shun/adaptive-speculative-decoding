# Directory Structure Guide

This document explains the organized structure of the Adaptive Speculative Decoding repository after refactoring.

## Overview

The repository has been reorganized into logical directories to improve maintainability and navigation:

```
adaptive-speculative-decoding/
├── 📁 docs/                           # All documentation
├── 📁 src/                            # Core source code  
├── 📁 experiments/                    # Experimental code
├── 📁 results/                        # Generated results
├── 📁 configs/                        # Configuration files
├── 📁 scripts/                        # Setup and utility scripts
├── 📁 tests/                          # Test suite
└── 📁 archive/                        # Archived/legacy files
```

## Detailed Structure

### 📁 docs/ - Documentation
```
docs/
├── papers/                            # Research papers
│   ├── FINAL_PAPER.md                # Main research paper
│   ├── ENHANCED_RESEARCH_PAPER.md    # Enhanced version
│   └── PAPER.md                      # Alternative version
├── guides/                            # User and developer guides
│   ├── EXPERIMENT_GUIDE.md           # How to run experiments
│   ├── GETTING_STARTED.md            # Quick start guide
│   └── RESEARCH_PROTOCOL.md          # Research methodology
├── summaries/                         # Result summaries
│   ├── EXECUTIVE_SUMMARY.md          # Executive summary
│   ├── EXPERIMENT_RESULTS_SUMMARY.md # Detailed results
│   ├── RESEARCH_SUMMARY.md           # Research overview
│   └── REFACTORING_SUMMARY.md        # Code refactoring notes
└── README_ACADEMIC.md                 # Academic documentation
```

### 📁 experiments/ - Experimental Code
```
experiments/
├── final/                             # Production-ready experiments
│   ├── run_final_real_experiments.py # Main experiment runner
│   ├── simple_experiments.py         # Core empirical validation
│   ├── simple_theory_demo.py         # Theoretical demonstration
│   └── setup_datasets.py             # Dataset preparation
├── development/                       # Development experiments
│   ├── run_7b_experiments.py         # 7B model testing
│   ├── test_7b_only.py              # Single model validation
│   ├── test_single_model.py         # Basic model testing
│   └── run_progressive_experiments.py # Progressive testing
├── scripts/                          # Analysis and utility scripts
│   ├── analyze_baseline_comparison.py
│   ├── create_final_visualizations.py
│   ├── generate_training_data_large_scale.py
│   └── [other analysis scripts]
└── legacy/                           # Legacy experiment files
    ├── run_complete_experiments.sh
    └── [old experiment files]
```

### 📁 results/ - Generated Results
```
results/
├── figures/                          # Research figures
│   ├── paper_results.png            # Main 4-panel figure
│   ├── paper_results.pdf            # PDF version
│   ├── theoretical_results_simple.png # Theory validation
│   └── [other generated figures]
├── data/                             # Experimental data
│   └── [JSON result files]
└── final/                            # Final experiment results
    └── [comprehensive result files]
```

### 📁 src/ - Source Code
```
src/
├── algorithms/                       # Core algorithms
│   ├── dp_solver.py                 # Dynamic programming solver
│   └── optimizer.py                 # Optimization algorithms
├── models/                          # Model implementations
│   ├── predictor.py                 # Quality predictor
│   ├── stage.py                     # Stage model wrapper
│   └── enhanced_predictor.py        # Enhanced predictor
├── serving/                         # Inference pipeline
│   ├── pipeline.py                  # Main inference pipeline
│   ├── server.py                    # Server implementation
│   └── cache_manager.py             # Memory management
├── core/                            # Core interfaces and types
│   ├── interfaces.py                # Abstract interfaces
│   ├── types.py                     # Type definitions
│   └── exceptions.py                # Custom exceptions
├── config/                          # Configuration management
│   ├── base.py                      # Base configuration
│   ├── model_config.py              # Model configurations
│   └── [other config files]
├── utils/                           # Utility functions
│   ├── logging_utils.py             # Logging utilities
│   ├── timing_utils.py              # Performance timing
│   └── validation_utils.py          # Input validation
└── training/                        # Training infrastructure
    └── generate_training_data.py    # Training data generation
```

## File Categories

### 🎯 Core Research Files
- `docs/papers/FINAL_PAPER.md` - Main research paper
- `experiments/final/simple_experiments.py` - Core experiments
- `experiments/final/simple_theory_demo.py` - Theoretical validation
- `results/figures/paper_results.png` - Main research figures

### 🔧 Implementation Files
- `src/algorithms/` - Core algorithmic implementations
- `src/serving/pipeline.py` - Production inference pipeline
- `src/models/predictor.py` - Quality prediction model

### 🧪 Experimental Files
- `experiments/final/` - Production experiments
- `experiments/development/` - Development testing
- `experiments/scripts/` - Analysis utilities

### 📊 Results Files
- `results/figures/` - Generated research figures
- `results/data/` - Experimental data outputs
- `results/final/` - Comprehensive results

## Navigation Guide

### For Researchers
1. Start with `docs/papers/FINAL_PAPER.md` for the complete research
2. Review `results/figures/` for key visualizations
3. Examine `experiments/final/` for core experimental code

### For Developers  
1. Explore `src/` for implementation details
2. Check `experiments/development/` for testing code
3. Use `configs/` for system configuration

### For Users
1. Begin with `README.md` for overview
2. Follow `docs/guides/GETTING_STARTED.md` for setup
3. Run experiments from `experiments/final/`

## Benefits of This Structure

1. **Clear Separation**: Documentation, code, experiments, and results are logically separated
2. **Easy Navigation**: Related files are grouped together
3. **Scalability**: New files can be easily categorized
4. **Maintainability**: Reduces clutter and improves code organization
5. **Professionalism**: Follows best practices for research repositories

## Migration Notes

All files have been moved to appropriate directories while maintaining their functionality. The new structure provides:

- Better organization for conference submission
- Easier code review and maintenance  
- Clear separation of concerns
- Professional repository appearance
- Improved reproducibility for researchers

This reorganization makes the repository suitable for academic publication and open-source distribution.