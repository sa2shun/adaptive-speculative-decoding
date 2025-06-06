#!/usr/bin/env python3
"""
Generate large-scale training data for quality predictor
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic training data for quality predictor"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Categories of prompts with different complexity levels
        self.prompt_templates = {
            "factual_simple": [
                "What is the capital of {}?",
                "Who wrote {}?",
                "When was {} invented?",
                "What color is {}?",
                "How many {} are in {}?"
            ],
            "factual_medium": [
                "Explain the historical significance of {}.",
                "What are the main causes of {}?",
                "Describe the process of {}.",
                "What are the benefits and drawbacks of {}?",
                "How does {} affect {}?"
            ],
            "analytical_complex": [
                "Analyze the relationship between {} and {} considering multiple perspectives.",
                "Evaluate the long-term implications of {} on {}.",
                "Compare and contrast {} with {} across different dimensions.",
                "Critically assess the effectiveness of {} in addressing {}.",
                "Synthesize information about {} to develop a comprehensive framework for {}."
            ],
            "programming_simple": [
                "Write a function to {}.",
                "How do you {} in Python?",
                "What is the syntax for {}?",
                "Fix this bug in {}.",
                "Explain how {} works."
            ],
            "programming_complex": [
                "Design a distributed system architecture for {}.",
                "Implement a {} algorithm with comprehensive error handling and optimization.",
                "Create a machine learning pipeline for {} with proper validation and monitoring.",
                "Develop a secure API for {} with authentication, rate limiting, and data validation.",
                "Build a scalable {} solution considering performance, maintainability, and testing."
            ],
            "creative_medium": [
                "Write a short story about {}.",
                "Create a poem describing {}.",
                "Design a character who {}.",
                "Imagine a world where {}.",
                "Compose a dialogue between {} and {}."
            ],
            "creative_complex": [
                "Develop a detailed narrative exploring the psychological journey of {}.",
                "Create an intricate world-building framework for a story about {}.",
                "Design a multi-layered character arc that demonstrates {}.",
                "Compose a thought-provoking piece that challenges conventional views on {}.",
                "Craft a complex narrative structure that weaves together themes of {} and {}."
            ]
        }
        
        # Vocabulary for filling templates
        self.vocabulary = {
            "countries": ["France", "Japan", "Brazil", "Egypt", "Australia"],
            "books": ["Pride and Prejudice", "1984", "The Great Gatsby", "To Kill a Mockingbird"],
            "technologies": ["the internet", "smartphones", "solar panels", "electric cars"],
            "colors": ["the sky", "grass", "fire", "snow"],
            "objects": ["stars", "books", "cars", "trees", "people"],
            "containers": ["a library", "a city", "a forest", "a galaxy"],
            "concepts": ["democracy", "climate change", "artificial intelligence", "globalization"],
            "processes": ["photosynthesis", "evolution", "market economics", "neural learning"],
            "algorithms": ["binary search", "quicksort", "neural network", "decision tree"],
            "systems": ["chat application", "e-commerce platform", "data pipeline", "microservice"],
            "characters": ["a detective", "a scientist", "an artist", "a traveler"],
            "themes": ["justice", "love", "identity", "power", "freedom"]
        }
    
    def generate_prompt(self, category: str, complexity: str) -> str:
        """Generate a prompt from a specific category and complexity"""
        template_key = f"{category}_{complexity}"
        
        if template_key not in self.prompt_templates:
            # Fallback to available keys
            available_keys = [k for k in self.prompt_templates.keys() if category in k]
            if not available_keys:
                template_key = "factual_simple"
            else:
                template_key = random.choice(available_keys)
        
        template = random.choice(self.prompt_templates[template_key])
        
        # Fill template with vocabulary
        placeholders = template.count('{}')
        if placeholders > 0:
            # Choose appropriate vocabulary
            vocab_keys = list(self.vocabulary.keys())
            chosen_words = []
            
            for _ in range(placeholders):
                key = random.choice(vocab_keys)
                word = random.choice(self.vocabulary[key])
                chosen_words.append(word)
            
            try:
                prompt = template.format(*chosen_words)
            except:
                # Fallback for format issues
                prompt = template.replace('{}', random.choice(self.vocabulary['concepts']))
        else:
            prompt = template
        
        return prompt
    
    def calculate_features(self, prompt: str, stage_id: int) -> List[float]:
        """Calculate features for a given prompt and stage"""
        # Feature 1: Input entropy (simplified)
        words = prompt.split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        if len(words) > 0:
            entropy = -sum((count/len(words)) * np.log2(count/len(words)) 
                          for count in word_counts.values() if count > 0)
        else:
            entropy = 0.0
        
        # Feature 2: Length ratio
        length_ratio = min(len(prompt) / 2048.0, 1.0)
        
        # Feature 3: Vocabulary diversity
        vocab_diversity = len(set(words)) / len(words) if words else 0
        
        # Feature 4: Average word length
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Feature 5-8: Stage encoding (one-hot)
        stage_features = [0.0] * 4
        if 0 <= stage_id < 4:
            stage_features[stage_id] = 1.0
        
        return [entropy, length_ratio, vocab_diversity, avg_word_length] + stage_features
    
    def simulate_acceptance_probability(
        self, 
        prompt: str, 
        stage_id: int, 
        complexity_category: str
    ) -> float:
        """Simulate acceptance probability based on prompt and stage"""
        
        # Base probabilities by complexity
        complexity_base_probs = {
            "simple": 0.9,
            "medium": 0.7, 
            "complex": 0.4
        }
        
        # Category adjustments
        category_adjustments = {
            "factual": 0.1,
            "analytical": -0.1,
            "programming": -0.15,
            "creative": -0.05
        }
        
        # Extract complexity and category
        complexity = complexity_category.split('_')[-1]  # simple, medium, complex
        category = complexity_category.split('_')[0]     # factual, analytical, etc.
        
        base_prob = complexity_base_probs.get(complexity, 0.7)
        category_adj = category_adjustments.get(category, 0.0)
        
        # Stage penalty (later stages are less likely to accept)
        stage_penalty = stage_id * 0.15
        
        # Length penalty
        length_penalty = max(0, (len(prompt) - 100) / 1000)
        
        # Calculate final probability
        prob = base_prob + category_adj - stage_penalty - length_penalty
        
        # Add some noise
        noise = np.random.normal(0, 0.05)
        prob += noise
        
        # Clamp to valid range
        prob = max(0.05, min(0.95, prob))
        
        return prob
    
    def generate_training_sample(self) -> Dict[str, Any]:
        """Generate a single training sample"""
        
        # Choose complexity and category
        categories = ["factual", "analytical", "programming", "creative"]
        complexities = ["simple", "medium", "complex"]
        
        category = random.choice(categories)
        complexity = random.choice(complexities)
        
        # Generate prompt
        prompt = self.generate_prompt(category, complexity)
        
        # Choose stage
        stage_id = random.randint(0, 3)
        
        # Calculate features
        features = self.calculate_features(prompt, stage_id)
        
        # Simulate acceptance probability
        complexity_category = f"{category}_{complexity}"
        acceptance_prob = self.simulate_acceptance_probability(
            prompt, stage_id, complexity_category
        )
        
        return {
            "prompt": prompt,
            "stage_id": stage_id,
            "category": category,
            "complexity": complexity,
            "features": features,
            "acceptance_probability": acceptance_prob,
            "prompt_length": len(prompt),
            "word_count": len(prompt.split())
        }
    
    def generate_dataset(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate a complete training dataset"""
        
        logger.info(f"Generating {num_samples} training samples...")
        
        dataset = []
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            sample = self.generate_training_sample()
            dataset.append(sample)
        
        logger.info(f"Generated {len(dataset)} samples")
        
        return dataset


def analyze_dataset(dataset: List[Dict[str, Any]]) -> None:
    """Analyze the generated dataset"""
    
    df = pd.DataFrame(dataset)
    
    print("\n📊 Dataset Analysis")
    print("=" * 50)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Average prompt length: {df['prompt_length'].mean():.1f} chars")
    print(f"Average word count: {df['word_count'].mean():.1f} words")
    
    print(f"\n🏷️  Category Distribution:")
    for category, count in df['category'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Complexity Distribution:")
    for complexity, count in df['complexity'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"   {complexity}: {count} ({percentage:.1f}%)")
    
    print(f"\n⚙️  Stage Distribution:")
    for stage, count in df['stage_id'].value_counts().sort_index().items():
        percentage = (count / len(df)) * 100
        print(f"   Stage {stage}: {count} ({percentage:.1f}%)")
    
    print(f"\n📈 Acceptance Probability Stats:")
    print(f"   Mean: {df['acceptance_probability'].mean():.3f}")
    print(f"   Std:  {df['acceptance_probability'].std():.3f}")
    print(f"   Min:  {df['acceptance_probability'].min():.3f}")
    print(f"   Max:  {df['acceptance_probability'].max():.3f}")


def save_dataset(dataset: List[Dict[str, Any]], output_dir: str) -> None:
    """Save dataset in multiple formats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path / "training_data.json"
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved JSON dataset: {json_path}")
    
    # Save as CSV
    df = pd.DataFrame(dataset)
    csv_path = output_path / "training_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV dataset: {csv_path}")
    
    # Save features and labels separately
    features = np.array([sample['features'] for sample in dataset])
    labels = np.array([sample['acceptance_probability'] for sample in dataset])
    
    np.save(output_path / "features.npy", features)
    np.save(output_path / "labels.npy", labels)
    logger.info(f"Saved features and labels: {output_path}")
    
    # Save metadata
    metadata = {
        "num_samples": len(dataset),
        "feature_dimension": len(dataset[0]['features']),
        "categories": list(set(sample['category'] for sample in dataset)),
        "complexities": list(set(sample['complexity'] for sample in dataset)),
        "feature_names": [
            "entropy", "length_ratio", "vocab_diversity", "avg_word_length",
            "stage_0", "stage_1", "stage_2", "stage_3"
        ]
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {output_path / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for quality predictor")
    parser.add_argument("--num-samples", type=int, default=100000, 
                       help="Number of training samples to generate")
    parser.add_argument("--output-dir", default="/raid/sasaki/adaptive-sd-training-data",
                       help="Output directory for training data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze", action="store_true", help="Analyze generated dataset")
    
    args = parser.parse_args()
    
    print("🚀 Large-Scale Training Data Generation")
    print("=" * 80)
    print(f"Samples: {args.num_samples:,}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    
    # Generate dataset
    generator = SyntheticDataGenerator(seed=args.seed)
    dataset = generator.generate_dataset(args.num_samples)
    
    # Analyze if requested
    if args.analyze:
        analyze_dataset(dataset)
    
    # Save dataset
    save_dataset(dataset, args.output_dir)
    
    print(f"\n✅ Training data generation complete!")
    print(f"   {len(dataset):,} samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()