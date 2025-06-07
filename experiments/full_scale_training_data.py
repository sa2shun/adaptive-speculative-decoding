#!/usr/bin/env python3
"""
Generate 100K training samples with REAL Qwen2.5 model inference
NO simulation - research-grade quality data generation
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from tqdm import tqdm
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealModelTrainingDataGenerator:
    """Generate training data using actual Qwen2.5 models."""
    
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load all available models
        self.load_models()
        
        # Quality assessment criteria
        self.quality_criteria = {
            "coherence": 0.3,
            "relevance": 0.3, 
            "completeness": 0.2,
            "accuracy": 0.2
        }
    
    def load_models(self):
        """Load all Qwen2.5 models for data generation."""
        model_configs = {
            "7b": "qwen3-7b",
            "14b": "qwen3-14b", 
            "32b": "qwen3-32b",
            "72b": "qwen3-72b"
        }
        
        for size, path in model_configs.items():
            model_path = Path(self.model_base_path) / path
            if model_path.exists():
                logger.info(f"Loading {size} model from {model_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.tokenizers[size] = tokenizer
                    self.models[size] = model
                    logger.info(f"Successfully loaded {size} model")
                except Exception as e:
                    logger.error(f"Failed to load {size} model: {e}")
            else:
                logger.warning(f"Model path {model_path} does not exist")
    
    def generate_diverse_prompts(self, num_prompts: int) -> List[Dict]:
        """Generate diverse prompts across complexity levels."""
        prompts = []
        
        # Simple factual queries (30%)
        simple_templates = [
            "What is {}?",
            "Explain {} in simple terms.",
            "List the main features of {}.",
            "Define {}.",
            "Who is {}?"
        ]
        
        simple_topics = [
            "machine learning", "climate change", "photosynthesis", "democracy",
            "Einstein's theory of relativity", "quantum computing", "blockchain",
            "artificial intelligence", "renewable energy", "genetic engineering"
        ]
        
        # Medium complexity analytical queries (40%)
        medium_templates = [
            "Analyze the impact of {} on society.",
            "Compare {} and {} in terms of efficiency and cost.",
            "Explain the relationship between {} and {}.",
            "What are the advantages and disadvantages of {}?",
            "How does {} work and why is it important?"
        ]
        
        medium_topics = [
            ("artificial intelligence", "job market"),
            ("renewable energy", "fossil fuels"),
            ("social media", "mental health"),
            ("cryptocurrency", "traditional banking"),
            ("remote work", "office work")
        ]
        
        # Complex reasoning queries (30%)
        complex_templates = [
            "Critically evaluate the long-term implications of {} considering economic, social, and environmental factors.",
            "Develop a comprehensive strategy for addressing {} while balancing competing interests of different stakeholders.",
            "Synthesize current research on {} to propose a novel theoretical framework.",
            "Analyze the ethical implications of {} and propose guidelines for responsible implementation.",
            "Examine the historical evolution of {} and predict future developments based on current trends."
        ]
        
        complex_topics = [
            "gene editing technology", "artificial general intelligence",
            "climate intervention technologies", "universal basic income",
            "quantum internet", "space colonization", "brain-computer interfaces"
        ]
        
        # Generate prompts
        for _ in range(num_prompts // 3):
            # Simple
            template = random.choice(simple_templates)
            topic = random.choice(simple_topics)
            prompts.append({
                "prompt": template.format(topic),
                "complexity": "simple",
                "expected_difficulty": 0.3
            })
            
            # Medium
            template = random.choice(medium_templates)
            if "{}" in template and template.count("{}") == 2:
                topic1, topic2 = random.choice(medium_topics)
                prompt_text = template.format(topic1, topic2)
            else:
                topic = random.choice([t[0] for t in medium_topics])
                prompt_text = template.format(topic)
            
            prompts.append({
                "prompt": prompt_text,
                "complexity": "medium", 
                "expected_difficulty": 0.6
            })
            
            # Complex
            template = random.choice(complex_templates)
            topic = random.choice(complex_topics)
            prompts.append({
                "prompt": template.format(topic),
                "complexity": "complex",
                "expected_difficulty": 0.9
            })
        
        return prompts
    
    def generate_with_model(self, prompt: str, model_size: str, max_tokens: int = 512) -> Tuple[str, float]:
        """Generate response with specific model and measure quality."""
        if model_size not in self.models:
            return "", 0.0
        
        model = self.models[model_size]
        tokenizer = self.tokenizers[model_size]
        
        try:
            # Measure inference time for cost modeling
            start_time = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            inference_time = time.time() - start_time
            
            # Estimate quality based on response characteristics
            quality = self.estimate_quality(prompt, response, model_size)
            
            return response, quality, inference_time
            
        except Exception as e:
            logger.error(f"Error generating with {model_size}: {e}")
            return "", 0.0, 0.0
    
    def estimate_quality(self, prompt: str, response: str, model_size: str) -> float:
        """Estimate response quality using heuristics."""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        quality_score = 0.0
        
        # Length-based scoring (reasonable length indicates effort)
        response_length = len(response.split())
        if 20 <= response_length <= 200:
            quality_score += 0.3
        elif 10 <= response_length < 20 or 200 < response_length <= 300:
            quality_score += 0.2
        else:
            quality_score += 0.1
        
        # Coherence (no repetition, proper structure)
        sentences = response.split('.')
        if len(set(sentences)) > len(sentences) * 0.8:  # Low repetition
            quality_score += 0.2
        
        # Relevance (keywords from prompt appear in response)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        quality_score += min(0.3, overlap)
        
        # Model size bias (larger models generally better)
        model_bonus = {"7b": 0.0, "14b": 0.1, "32b": 0.15, "72b": 0.2}.get(model_size, 0.0)
        quality_score += model_bonus
        
        return min(1.0, quality_score)
    
    def generate_training_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate complete training dataset."""
        logger.info(f"Starting generation of {num_samples} training samples")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate diverse prompts
        logger.info("Generating diverse prompts...")
        prompts = self.generate_diverse_prompts(num_samples)
        random.shuffle(prompts)
        
        training_data = []
        model_sizes = list(self.models.keys())
        
        for i, prompt_info in enumerate(tqdm(prompts, desc="Generating training data")):
            prompt = prompt_info["prompt"]
            complexity = prompt_info["complexity"]
            expected_difficulty = prompt_info["expected_difficulty"]
            
            # Generate responses with all available models
            sample = {
                "prompt": prompt,
                "complexity": complexity,
                "expected_difficulty": expected_difficulty,
                "responses": {},
                "quality_scores": {},
                "inference_times": {},
                "sample_id": i
            }
            
            for model_size in model_sizes:
                response, quality, inf_time = self.generate_with_model(prompt, model_size)
                sample["responses"][model_size] = response
                sample["quality_scores"][model_size] = quality
                sample["inference_times"][model_size] = inf_time
            
            training_data.append(sample)
            
            # Save intermediate results every 1000 samples
            if (i + 1) % 1000 == 0:
                intermediate_file = output_path / f"training_data_batch_{i//1000 + 1}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump(training_data[-1000:], f, indent=2)
                logger.info(f"Saved batch {i//1000 + 1}, processed {i+1}/{len(prompts)} samples")
        
        # Save complete dataset
        final_file = output_path / "complete_training_data.json"
        with open(final_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Generate statistics
        stats = self.compute_dataset_statistics(training_data)
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training data generation complete!")
        logger.info(f"Generated {len(training_data)} samples")
        logger.info(f"Data saved to {output_path}")
        
        return {
            "training_data": training_data,
            "statistics": stats,
            "output_path": str(output_path)
        }
    
    def compute_dataset_statistics(self, training_data: List[Dict]) -> Dict:
        """Compute statistics for the generated dataset."""
        stats = {
            "total_samples": len(training_data),
            "complexity_distribution": {},
            "quality_statistics": {},
            "model_performance": {}
        }
        
        # Complexity distribution
        complexity_counts = {}
        for sample in training_data:
            complexity = sample["complexity"]
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        stats["complexity_distribution"] = complexity_counts
        
        # Quality statistics per model
        for model_size in self.models.keys():
            qualities = [sample["quality_scores"].get(model_size, 0) for sample in training_data]
            stats["quality_statistics"][model_size] = {
                "mean": np.mean(qualities),
                "std": np.std(qualities),
                "min": np.min(qualities),
                "max": np.max(qualities)
            }
        
        # Model performance comparison
        for model_size in self.models.keys():
            times = [sample["inference_times"].get(model_size, 0) for sample in training_data]
            stats["model_performance"][model_size] = {
                "avg_inference_time": np.mean(times),
                "total_samples": len([t for t in times if t > 0])
            }
        
        return stats


def main():
    """Run full-scale training data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large-scale training data")
    parser.add_argument("--model-path", default="/raid/$USER/adaptive-sd-models", 
                       help="Path to Qwen2.5 models")
    parser.add_argument("--output-path", default="/raid/$USER/adaptive-sd-training-data",
                       help="Output directory for training data")
    parser.add_argument("--num-samples", type=int, default=100000,
                       help="Number of training samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Expand environment variables
    model_path = os.path.expandvars(args.model_path)
    output_path = os.path.expandvars(args.output_path)
    
    logger.info("=== FULL-SCALE TRAINING DATA GENERATION ===")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Seed: {args.seed}")
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize generator
    generator = RealModelTrainingDataGenerator(model_path)
    
    if not generator.models:
        logger.error("No models loaded! Check model path.")
        return
    
    logger.info(f"Loaded models: {list(generator.models.keys())}")
    
    # Generate training data
    result = generator.generate_training_dataset(args.num_samples, output_path)
    
    logger.info("=== GENERATION COMPLETE ===")
    logger.info(f"Statistics: {result['statistics']}")


if __name__ == "__main__":
    main()