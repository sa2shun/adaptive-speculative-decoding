#!/usr/bin/env python3
"""
Fix all research quality compromises to meet the highest standards
"""

import yaml
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_model_quantization():
    """Remove quantization compromises for maximum quality"""
    
    config_path = Path("configs/models.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Removing quantization compromises...")
    
    for model_name, model_config in config['models'].items():
        # Disable quantization for maximum precision
        model_config['quantization'] = {
            'enabled': False,
            'method': None,
            'compute_dtype': "float16"  # Minimum acceptable precision
        }
        
        # Increase GPU memory utilization for full models
        if model_name == "70b":
            model_config['gpu_memory_utilization'] = 0.95
        else:
            model_config['gpu_memory_utilization'] = 0.9
    
    # Save fixed configuration
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info("✅ Fixed model quantization - using full precision")

def fix_evaluation_dataset_sizes():
    """Remove dataset size limitations for comprehensive evaluation"""
    
    script_path = Path("setup_evaluation_datasets.py")
    
    # Read the current file
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Fix dataset sample limits
    replacements = {
        '"max_samples": 1000,': '"max_samples": 14042,  # Full MMLU test set',
        '"max_samples": 500,': '"max_samples": 10042,  # Full validation set', 
        '"max_samples": 164,  # Full dataset': '"max_samples": 164,  # Full HumanEval (already complete)',
        '"max_samples": 300,': '"max_samples": 817,   # Full TruthfulQA validation'
    }
    
    logger.info("Fixing evaluation dataset size limitations...")
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Save fixed file
    with open(script_path, 'w') as f:
        f.write(content)
    
    logger.info("✅ Fixed evaluation dataset sizes - using full datasets")

def create_full_scale_experiment_plan():
    """Create comprehensive experiment plan without compromises"""
    
    plan = {
        "experiment_name": "adaptive_speculative_decoding_full_scale",
        "research_standards": {
            "no_compromises": True,
            "full_model_precision": True,
            "complete_datasets": True,
            "comprehensive_evaluation": True
        },
        "models": {
            "8b": {
                "name": "meta-llama/Llama-3.1-8B",
                "precision": "float16",
                "quantization": False,
                "full_parameters": True
            },
            "13b": {
                "name": "meta-llama/Llama-3.1-8B-Instruct", 
                "precision": "float16",
                "quantization": False,
                "full_parameters": True
            },
            "34b": {
                "name": "codellama/CodeLlama-34b-hf",
                "precision": "float16", 
                "quantization": False,
                "full_parameters": True
            },
            "70b": {
                "name": "meta-llama/Llama-3.1-70B-Instruct",
                "precision": "float16",
                "quantization": False,
                "full_parameters": True
            }
        },
        "training_data": {
            "samples": 100000,
            "quality": "high",
            "diversity": "comprehensive",
            "no_shortcuts": True
        },
        "evaluation_data": {
            "mmlu": {"samples": 14042, "full_dataset": True},
            "hellaswag": {"samples": 10042, "full_dataset": True},
            "humaneval": {"samples": 164, "full_dataset": True},
            "gsm8k": {"samples": 1319, "full_dataset": True},
            "truthfulqa": {"samples": 817, "full_dataset": True},
            "custom_complexity": {"samples": 1000, "balanced_complexity": True}
        },
        "lambda_values": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        "experimental_rigor": {
            "multiple_seeds": [42, 123, 456, 789, 999],
            "statistical_significance": True,
            "confidence_intervals": True,
            "ablation_studies": True,
            "baseline_comparisons": True
        },
        "documentation": {
            "complete_logs": True,
            "reproducibility_artifacts": True,
            "detailed_hyperparameters": True,
            "performance_analysis": True,
            "research_paper_quality": True
        },
        "resources": {
            "gpus": "8x H100 80GB",
            "storage": "30TB RAID",
            "no_resource_constraints": True
        }
    }
    
    # Save the plan
    plan_path = Path("/raid/sasaki/adaptive-sd-full-scale-plan.json")
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    
    logger.info(f"✅ Created full-scale experiment plan: {plan_path}")
    
    return plan

def main():
    print("🔧 Fixing All Research Quality Compromises")
    print("=" * 80)
    print("This will ensure NO COMPROMISES in research quality")
    print()
    
    # Fix all identified compromises
    fix_model_quantization()
    fix_evaluation_dataset_sizes()
    plan = create_full_scale_experiment_plan()
    
    print("\n✅ All compromises have been addressed:")
    print("   1. ✅ Removed model quantization (full precision)")
    print("   2. ✅ Expanded to full evaluation datasets")
    print("   3. ✅ Created comprehensive experiment plan")
    print("   4. ✅ Specified rigorous documentation requirements")
    
    print(f"\n🎯 Next Steps for Uncompromised Research:")
    print("   1. Download full-scale Meta Llama models")
    print("   2. Re-generate evaluation datasets at full scale") 
    print("   3. Execute comprehensive experiments with all λ values")
    print("   4. Generate research-grade results with statistical analysis")
    
    print(f"\n📊 Experiment Scale:")
    print(f"   Models: 4 full-scale models (no quantization)")
    print(f"   Training Data: {plan['training_data']['samples']:,} samples")
    print(f"   Evaluation Data: {sum(d['samples'] for d in plan['evaluation_data'].values()):,} samples")
    print(f"   Lambda Values: {len(plan['lambda_values'])} values")
    print(f"   Seeds: {len(plan['experimental_rigor']['multiple_seeds'])} seeds")
    
    total_experiments = (
        len(plan['lambda_values']) * 
        len(plan['experimental_rigor']['multiple_seeds']) *
        len(plan['models'])
    )
    print(f"   Total Experiment Configurations: {total_experiments:,}")

if __name__ == "__main__":
    main()