#!/usr/bin/env python3
"""
Download and prepare evaluation datasets for experiments.

This script downloads MMLU, HumanEval, and creates MT-Bench style questions.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_mmlu(output_dir: Path, max_samples: int = 2000) -> int:
    """Download and prepare MMLU dataset."""
    logger.info("Downloading MMLU dataset...")
    
    try:
        # Load MMLU test set
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        # Select subset
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Convert to our format
        mmlu_data = []
        for item in dataset:
            formatted = {
                "id": f"mmlu_{len(mmlu_data)}",
                "prompt": f"Question: {item['question']}\n\nChoices:\nA) {item['choices'][0]}\nB) {item['choices'][1]}\nC) {item['choices'][2]}\nD) {item['choices'][3]}\n\nAnswer:",
                "reference": item['choices'][item['answer']],
                "subject": item.get('subject', 'general'),
                "difficulty": "moderate"  # MMLU is generally moderate difficulty
            }
            mmlu_data.append(formatted)
        
        # Save
        output_file = output_dir / "mmlu_test.json"
        with open(output_file, 'w') as f:
            json.dump(mmlu_data, f, indent=2)
        
        logger.info(f"Saved {len(mmlu_data)} MMLU examples to {output_file}")
        return len(mmlu_data)
        
    except Exception as e:
        logger.error(f"Failed to download MMLU: {e}")
        return 0


def download_humaneval(output_dir: Path) -> int:
    """Download and prepare HumanEval dataset."""
    logger.info("Downloading HumanEval dataset...")
    
    try:
        # Load HumanEval
        dataset = load_dataset("openai_humaneval", split="test")
        
        # Convert to our format
        humaneval_data = []
        for item in dataset:
            formatted = {
                "id": f"humaneval_{item['task_id']}",
                "prompt": item['prompt'],
                "reference": item['canonical_solution'],
                "test_cases": item['test'],
                "difficulty": "complex"  # Programming tasks are complex
            }
            humaneval_data.append(formatted)
        
        # Save
        output_file = output_dir / "humaneval_test.json"
        with open(output_file, 'w') as f:
            json.dump(humaneval_data, f, indent=2)
        
        logger.info(f"Saved {len(humaneval_data)} HumanEval examples to {output_file}")
        return len(humaneval_data)
        
    except Exception as e:
        logger.error(f"Failed to download HumanEval: {e}")
        return 0


def create_mt_bench(output_dir: Path, num_examples: int = 100) -> int:
    """Create MT-Bench style multi-turn reasoning questions."""
    logger.info("Creating MT-Bench style dataset...")
    
    # MT-Bench style questions covering different categories
    templates = [
        # Writing
        {
            "category": "writing",
            "prompts": [
                "Write a persuasive email to your manager requesting a flexible work schedule. Explain the benefits and address potential concerns.",
                "Compose a short story (200 words) that begins with 'The last person on Earth sat alone in a room. There was a knock on the door.'",
                "Write a product review for a fictional time machine. Include pros, cons, and a rating."
            ]
        },
        # Math
        {
            "category": "math",
            "prompts": [
                "A train travels from City A to City B at 60 mph. The return trip at 40 mph takes 2 hours longer. What is the distance between the cities? Show your work.",
                "Explain the concept of derivatives to a high school student. Include a practical example.",
                "Solve this system of equations: 2x + 3y = 12 and x - y = 1. Explain each step."
            ]
        },
        # Reasoning
        {
            "category": "reasoning", 
            "prompts": [
                "You have 3 boxes: one contains only apples, one only oranges, and one both. All boxes are mislabeled. You can pick one fruit from one box. How do you determine the correct labels?",
                "Analyze this argument: 'All birds can fly. Penguins are birds. Therefore, penguins can fly.' What's wrong with this reasoning?",
                "Design an experiment to test whether plants grow better with music. Include controls and variables."
            ]
        },
        # Coding
        {
            "category": "coding",
            "prompts": [
                "Write a Python function to find the second largest element in a list without using built-in sorting. Handle edge cases.",
                "Explain the difference between shallow and deep copying in Python. Provide examples.",
                "Debug this code: `def factorial(n): return n * factorial(n-1)`. What's missing?"
            ]
        },
        # Knowledge
        {
            "category": "knowledge",
            "prompts": [
                "Explain the process of photosynthesis in simple terms. What are the inputs and outputs?",
                "Compare and contrast the American and French Revolutions. List 3 similarities and 3 differences.",
                "How does a blockchain work? Explain it as if to someone who has never heard of cryptocurrency."
            ]
        }
    ]
    
    # Generate dataset
    mt_bench_data = []
    example_id = 0
    
    for _ in range(num_examples // len(templates)):
        for category_data in templates:
            for prompt in category_data["prompts"]:
                formatted = {
                    "id": f"mt_bench_{example_id}",
                    "prompt": prompt,
                    "category": category_data["category"],
                    "difficulty": "complex" if category_data["category"] in ["coding", "math"] else "moderate",
                    "reference": None  # MT-Bench typically doesn't have references
                }
                mt_bench_data.append(formatted)
                example_id += 1
                
                if len(mt_bench_data) >= num_examples:
                    break
            if len(mt_bench_data) >= num_examples:
                break
    
    # Save
    output_file = output_dir / "mt_bench_test.json"
    with open(output_file, 'w') as f:
        json.dump(mt_bench_data, f, indent=2)
    
    logger.info(f"Saved {len(mt_bench_data)} MT-Bench examples to {output_file}")
    return len(mt_bench_data)


def create_simple_qa(output_dir: Path, num_examples: int = 500) -> int:
    """Create simple Q&A pairs for testing easy queries."""
    logger.info("Creating simple Q&A dataset...")
    
    simple_qa = [
        {"q": "What is the capital of France?", "a": "Paris"},
        {"q": "What is 2 + 2?", "a": "4"},
        {"q": "Who wrote Romeo and Juliet?", "a": "William Shakespeare"},
        {"q": "What color is the sky?", "a": "Blue"},
        {"q": "How many days are in a week?", "a": "Seven"},
        {"q": "What is the largest planet in our solar system?", "a": "Jupiter"},
        {"q": "What is water made of?", "a": "Hydrogen and oxygen (H2O)"},
        {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci"},
        {"q": "What is the speed of light?", "a": "299,792,458 meters per second"},
        {"q": "How many continents are there?", "a": "Seven"}
    ]
    
    # Expand dataset by creating variations
    simple_data = []
    for i in range(num_examples):
        base = simple_qa[i % len(simple_qa)]
        formatted = {
            "id": f"simple_{i}",
            "prompt": base["q"],
            "reference": base["a"],
            "difficulty": "simple"
        }
        simple_data.append(formatted)
    
    # Save
    output_file = output_dir / "simple_qa_test.json"
    with open(output_file, 'w') as f:
        json.dump(simple_data, f, indent=2)
    
    logger.info(f"Saved {len(simple_data)} simple Q&A examples to {output_file}")
    return len(simple_data)


def main():
    parser = argparse.ArgumentParser(description="Setup evaluation datasets")
    parser.add_argument("--output-dir", type=str,
                       default=f"/raid/{os.environ.get('USER', 'sasaki')}/datasets",
                       help="Output directory for datasets")
    parser.add_argument("--mmlu-samples", type=int, default=2000,
                       help="Number of MMLU samples to download")
    parser.add_argument("--mt-bench-samples", type=int, default=100,
                       help="Number of MT-Bench style samples to create")
    parser.add_argument("--simple-qa-samples", type=int, default=500,
                       help="Number of simple Q&A samples to create")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setting up datasets in: {output_dir}")
    
    # Download/create each dataset
    total_samples = 0
    
    # MMLU
    mmlu_count = download_mmlu(output_dir, args.mmlu_samples)
    total_samples += mmlu_count
    
    # HumanEval
    humaneval_count = download_humaneval(output_dir)
    total_samples += humaneval_count
    
    # MT-Bench style
    mt_bench_count = create_mt_bench(output_dir, args.mt_bench_samples)
    total_samples += mt_bench_count
    
    # Simple Q&A
    simple_count = create_simple_qa(output_dir, args.simple_qa_samples)
    total_samples += simple_count
    
    # Create summary
    summary = {
        "total_samples": total_samples,
        "datasets": {
            "mmlu": {"count": mmlu_count, "file": "mmlu_test.json"},
            "humaneval": {"count": humaneval_count, "file": "humaneval_test.json"},
            "mt_bench": {"count": mt_bench_count, "file": "mt_bench_test.json"},
            "simple_qa": {"count": simple_count, "file": "simple_qa_test.json"}
        },
        "difficulty_distribution": {
            "simple": simple_count,
            "moderate": mmlu_count + (mt_bench_count // 2),
            "complex": humaneval_count + (mt_bench_count // 2)
        }
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("DATASET SETUP COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()