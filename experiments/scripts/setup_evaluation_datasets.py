#!/usr/bin/env python3
"""
Setup evaluation datasets for adaptive speculative decoding experiments
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import logging
from datasets import load_dataset
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationDatasetManager:
    """Manage evaluation datasets for adaptive speculative decoding"""
    
    def __init__(self, data_dir: str = "/raid/sasaki/adaptive-sd-eval-data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define datasets to download
        self.datasets_config = {
            "mmlu": {
                "huggingface_name": "cais/mmlu",
                "splits": ["test"],
                "max_samples": 14042,  # Full MMLU test set
                "complexity": "medium",
                "category": "knowledge"
            },
            "hellaswag": {
                "huggingface_name": "Rowan/hellaswag", 
                "splits": ["validation"],
                "max_samples": 10042,  # Full validation set
                "complexity": "medium",
                "category": "reasoning"
            },
            "humaneval": {
                "huggingface_name": "openai_humaneval",
                "splits": ["test"],
                "max_samples": 164,  # Full HumanEval (already complete)
                "complexity": "high",
                "category": "programming"
            },
            "gsm8k": {
                "huggingface_name": "gsm8k",
                "splits": ["test"],
                "max_samples": 10042,  # Full validation set
                "complexity": "medium",
                "category": "math"
            },
            "truthfulqa": {
                "huggingface_name": "truthful_qa",
                "splits": ["validation"],
                "max_samples": 817,   # Full TruthfulQA validation
                "complexity": "high",
                "category": "truthfulness"
            }
        }
    
    def download_dataset(self, dataset_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Download and process a specific dataset"""
        
        logger.info(f"Downloading {dataset_name}...")
        
        try:
            # Special handling for different datasets
            if dataset_name == "mmlu":
                dataset = load_dataset(config["huggingface_name"], "all")
                split_data = dataset["test"]
            elif dataset_name == "hellaswag":
                dataset = load_dataset(config["huggingface_name"])
                split_data = dataset["validation"]
            elif dataset_name == "humaneval":
                dataset = load_dataset(config["huggingface_name"])
                split_data = dataset["test"]
            elif dataset_name == "gsm8k":
                dataset = load_dataset(config["huggingface_name"], "main")
                split_data = dataset["test"]
            elif dataset_name == "truthfulqa":
                dataset = load_dataset(config["huggingface_name"], "generation")
                split_data = dataset["validation"]
            else:
                logger.error(f"Unknown dataset: {dataset_name}")
                return []
            
            # Process samples
            samples = []
            max_samples = min(config["max_samples"], len(split_data))
            
            # Random sampling if dataset is larger than max_samples
            if len(split_data) > max_samples:
                indices = random.sample(range(len(split_data)), max_samples)
                selected_data = [split_data[i] for i in indices]
            else:
                selected_data = split_data
            
            for i, item in enumerate(tqdm(selected_data, desc=f"Processing {dataset_name}")):
                sample = self._process_sample(dataset_name, item, i)
                if sample:
                    samples.append(sample)
            
            logger.info(f"Processed {len(samples)} samples from {dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return []
    
    def _process_sample(self, dataset_name: str, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single sample based on dataset type"""
        
        try:
            if dataset_name == "mmlu":
                prompt = f"Question: {item['question']}\\n"
                prompt += f"A) {item['choices'][0]}\\n"
                prompt += f"B) {item['choices'][1]}\\n" 
                prompt += f"C) {item['choices'][2]}\\n"
                prompt += f"D) {item['choices'][3]}\\n"
                prompt += "Answer:"
                
                return {
                    "id": f"mmlu_{index}",
                    "prompt": prompt,
                    "expected_answer": item['choices'][item['answer']],
                    "category": "multiple_choice",
                    "complexity": "medium",
                    "subject": item.get('subject', 'unknown'),
                    "dataset": "mmlu"
                }
                
            elif dataset_name == "hellaswag":
                prompt = f"Context: {item['ctx']}\\n"
                prompt += "Complete the scenario:"
                
                return {
                    "id": f"hellaswag_{index}",
                    "prompt": prompt,
                    "expected_answer": item['endings'][int(item['label'])],
                    "category": "completion",
                    "complexity": "medium",
                    "dataset": "hellaswag"
                }
                
            elif dataset_name == "humaneval":
                prompt = item['prompt']
                
                return {
                    "id": f"humaneval_{index}",
                    "prompt": prompt,
                    "expected_answer": item['canonical_solution'],
                    "category": "programming",
                    "complexity": "high",
                    "test_cases": item.get('test', ''),
                    "entry_point": item.get('entry_point', ''),
                    "dataset": "humaneval"
                }
                
            elif dataset_name == "gsm8k":
                prompt = f"Problem: {item['question']}\\n"
                prompt += "Solution:"
                
                return {
                    "id": f"gsm8k_{index}",
                    "prompt": prompt,
                    "expected_answer": item['answer'],
                    "category": "math",
                    "complexity": "medium",
                    "dataset": "gsm8k"
                }
                
            elif dataset_name == "truthfulqa":
                prompt = f"Question: {item['question']}\\n"
                prompt += "Answer:"
                
                return {
                    "id": f"truthfulqa_{index}",
                    "prompt": prompt,
                    "expected_answer": item.get('best_answer', ''),
                    "category": "factual",
                    "complexity": "high",
                    "dataset": "truthfulqa"
                }
                
        except Exception as e:
            logger.warning(f"Failed to process sample {index} from {dataset_name}: {e}")
            return None
    
    def download_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Download all configured datasets"""
        
        all_datasets = {}
        
        for dataset_name, config in self.datasets_config.items():
            logger.info(f"\\n=== Downloading {dataset_name} ===")
            samples = self.download_dataset(dataset_name, config)
            
            if samples:
                all_datasets[dataset_name] = samples
                
                # Save individual dataset
                dataset_file = self.data_dir / f"{dataset_name}.json"
                with open(dataset_file, 'w') as f:
                    json.dump(samples, f, indent=2)
                logger.info(f"Saved {dataset_name} to {dataset_file}")
            
        return all_datasets
    
    def create_combined_dataset(self, all_datasets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Create a combined dataset from all individual datasets"""
        
        combined = []
        
        for dataset_name, samples in all_datasets.items():
            combined.extend(samples)
        
        # Shuffle the combined dataset
        random.shuffle(combined)
        
        # Save combined dataset
        combined_file = self.data_dir / "combined_evaluation.json"
        with open(combined_file, 'w') as f:
            json.dump(combined, f, indent=2)
        
        logger.info(f"Created combined dataset with {len(combined)} samples: {combined_file}")
        
        return combined
    
    def analyze_datasets(self, all_datasets: Dict[str, List[Dict[str, Any]]]):
        """Analyze the downloaded datasets"""
        
        print("\\n📊 Evaluation Datasets Analysis")
        print("=" * 60)
        
        total_samples = 0
        
        for dataset_name, samples in all_datasets.items():
            print(f"\\n{dataset_name.upper()}:")
            print(f"  Samples: {len(samples)}")
            
            if samples:
                # Analyze categories
                categories = {}
                complexities = {}
                
                for sample in samples:
                    cat = sample.get('category', 'unknown')
                    comp = sample.get('complexity', 'unknown')
                    
                    categories[cat] = categories.get(cat, 0) + 1
                    complexities[comp] = complexities.get(comp, 0) + 1
                
                print(f"  Categories: {dict(categories)}")
                print(f"  Complexities: {dict(complexities)}")
                
                # Sample prompt length analysis
                prompt_lengths = [len(sample['prompt']) for sample in samples]
                avg_length = sum(prompt_lengths) / len(prompt_lengths)
                print(f"  Avg prompt length: {avg_length:.1f} chars")
            
            total_samples += len(samples)
        
        print(f"\\n📈 Summary:")
        print(f"  Total datasets: {len(all_datasets)}")
        print(f"  Total samples: {total_samples}")
        
        # Create metadata
        metadata = {
            "total_datasets": len(all_datasets),
            "total_samples": total_samples,
            "datasets": {
                name: {
                    "sample_count": len(samples),
                    "config": self.datasets_config[name]
                }
                for name, samples in all_datasets.items()
            }
        }
        
        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {metadata_file}")


def create_custom_complexity_dataset(data_dir: str, num_samples: int = 300):
    """Create a custom dataset specifically for testing complexity levels"""
    
    logger.info(f"Creating custom complexity dataset with {num_samples} samples...")
    
    complexity_prompts = {
        "simple": [
            "What is 2 + 2?",
            "Name the capital of France.",
            "What color is the sky?",
            "How many days are in a week?",
            "Who wrote Romeo and Juliet?",
        ],
        "medium": [
            "Explain the process of photosynthesis.",
            "What are the main causes of climate change?",
            "Describe how a computer processor works.",
            "Analyze the themes in Shakespeare's Hamlet.",
            "Explain the concept of machine learning.",
        ],
        "high": [
            "Design a distributed system architecture for a real-time trading platform that can handle millions of transactions per second while maintaining ACID properties.",
            "Develop a comprehensive framework for evaluating the ethical implications of AI decision-making in healthcare, considering multiple stakeholder perspectives.",
            "Create a detailed mathematical model for predicting market volatility during economic crises, incorporating behavioral economics and network effects.",
            "Design an algorithm for optimizing resource allocation in smart cities while balancing environmental impact, cost efficiency, and citizen satisfaction.",
            "Analyze the complex interplay between quantum mechanics and general relativity in the context of black hole information paradox.",
        ]
    }
    
    custom_samples = []
    samples_per_complexity = num_samples // 3
    
    for complexity, prompts in complexity_prompts.items():
        for i in range(samples_per_complexity):
            prompt = random.choice(prompts)
            
            # Add some variation
            if complexity == "simple":
                variations = ["", " Please be concise.", " Give a brief answer."]
            elif complexity == "medium":
                variations = ["", " Provide a detailed explanation.", " Include examples."]
            else:
                variations = ["", " Consider multiple approaches.", " Provide a comprehensive analysis."]
            
            final_prompt = prompt + random.choice(variations)
            
            custom_samples.append({
                "id": f"custom_{complexity}_{i}",
                "prompt": final_prompt,
                "category": "custom",
                "complexity": complexity,
                "dataset": "custom_complexity"
            })
    
    # Save custom dataset
    custom_file = Path(data_dir) / "custom_complexity.json"
    with open(custom_file, 'w') as f:
        json.dump(custom_samples, f, indent=2)
    
    logger.info(f"Created custom complexity dataset: {custom_file}")
    
    return custom_samples


def main():
    parser = argparse.ArgumentParser(description="Setup evaluation datasets")
    parser.add_argument("--data-dir", default="/raid/sasaki/adaptive-sd-eval-data",
                       help="Directory to save evaluation datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-custom", action="store_true", 
                       help="Include custom complexity dataset")
    parser.add_argument("--custom-samples", type=int, default=300,
                       help="Number of custom samples to generate")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("🚀 Setting up Evaluation Datasets")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Random seed: {args.seed}")
    
    # Create dataset manager
    manager = EvaluationDatasetManager(args.data_dir)
    
    # Download all datasets
    all_datasets = manager.download_all_datasets()
    
    # Create custom complexity dataset if requested
    if args.include_custom:
        logger.info("\\n=== Creating Custom Complexity Dataset ===")
        custom_samples = create_custom_complexity_dataset(
            args.data_dir, args.custom_samples
        )
        all_datasets["custom_complexity"] = custom_samples
    
    # Create combined dataset
    if all_datasets:
        combined = manager.create_combined_dataset(all_datasets)
        
        # Analyze datasets
        manager.analyze_datasets(all_datasets)
        
        print(f"\\n✅ Evaluation datasets setup complete!")
        print(f"   {len(all_datasets)} datasets downloaded")
        print(f"   {len(combined)} total samples")
        print(f"   Saved to: {args.data_dir}")
    else:
        print("\\n❌ No datasets were successfully downloaded")


if __name__ == "__main__":
    main()