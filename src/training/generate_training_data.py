#!/usr/bin/env python3
"""
Generate training data for quality predictor using real model outputs.

This script runs different model sizes on dataset samples and computes
quality scores to create training data.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single training sample for quality predictor."""
    prompt: str
    stage_id: int
    model_output: str
    reference_output: str
    features: List[float]
    quality_score: float
    bleu_score: float
    generation_time: float
    prompt_tokens: int
    completion_tokens: int


class ModelInference:
    """Handle inference for a single model."""
    
    def __init__(self, model_size: str, model_path: str, device_ids: List[int]):
        self.model_size = model_size
        self.model_path = model_path
        self.device_ids = device_ids
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model and tokenizer."""
        logger.info(f"Loading {self.model_size} model from {self.model_path}")
        
        from transformers import BitsAndBytesConfig
        
        # Quantization config for INT8
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        
        # Device map
        if len(self.device_ids) > 1:
            device_map = "auto"
        else:
            device_map = {"": self.device_ids[0]}
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Loaded {self.model_size} model successfully")
    
    def generate(self, prompt: str, max_new_tokens: int = 128) -> Tuple[str, float, Dict]:
        """Generate text and return output with metadata."""
        import time
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        generated_ids = outputs.sequences[0][len(input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compute logprobs
        if outputs.scores:
            logprobs = []
            for i, score in enumerate(outputs.scores):
                probs = F.softmax(score[0], dim=-1)
                token_id = generated_ids[i]
                logprob = torch.log(probs[token_id]).item()
                logprobs.append(logprob)
        else:
            logprobs = []
        
        metadata = {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(generated_ids),
            "logprobs": logprobs,
            "generation_time": generation_time
        }
        
        return generated_text, generation_time, metadata


def extract_features(prompt: str, output: str, metadata: Dict, stage_id: int) -> List[float]:
    """Extract features for quality prediction."""
    features = []
    
    # 1. Prompt features
    prompt_words = prompt.split()
    features.append(len(prompt_words))  # Word count
    features.append(len(prompt))  # Character count
    
    # 2. Output features  
    output_words = output.split()
    features.append(len(output_words))  # Output word count
    features.append(len(output))  # Output character count
    
    # 3. Length ratio
    length_ratio = len(output_words) / max(len(prompt_words), 1)
    features.append(length_ratio)
    
    # 4. Logprob statistics
    logprobs = metadata.get("logprobs", [])
    if logprobs:
        features.append(np.mean(logprobs))  # Mean logprob
        features.append(np.std(logprobs))   # Std logprob
        features.append(np.min(logprobs))   # Min logprob
        features.append(np.percentile(logprobs, 25))  # Q1
        features.append(np.median(logprobs))  # Median
    else:
        features.extend([0.0] * 5)
    
    # 5. Entropy estimate (based on vocabulary diversity)
    unique_words = len(set(output_words))
    vocab_diversity = unique_words / max(len(output_words), 1)
    features.append(vocab_diversity)
    
    # 6. Stage information (one-hot encoding)
    stage_features = [0.0] * 4
    stage_features[stage_id] = 1.0
    features.extend(stage_features)
    
    # 7. Generation speed
    gen_time = metadata.get("generation_time", 1.0)
    tokens_per_sec = metadata.get("completion_tokens", 0) / max(gen_time, 0.001)
    features.append(tokens_per_sec)
    
    # 8. Prompt complexity indicators
    has_code = int("def " in prompt or "```" in prompt or "import " in prompt)
    has_math = int(any(c in prompt for c in "+=*/<>"))
    question_words = sum(1 for w in ["what", "why", "how", "when", "where", "which"] if w in prompt.lower())
    
    features.append(has_code)
    features.append(has_math)
    features.append(question_words)
    
    # Pad to 64 dimensions
    while len(features) < 64:
        features.append(0.0)
    
    return features[:64]  # Ensure exactly 64 features


def compute_quality_score(output: str, reference: str, bleu_metric) -> Tuple[float, float]:
    """Compute quality score using BLEU."""
    try:
        # Compute BLEU score
        bleu_result = bleu_metric.compute(
            predictions=[output],
            references=[[reference]]
        )
        bleu_score = bleu_result["bleu"]
        
        # Convert to binary label (threshold at 0.7)
        quality_label = 1.0 if bleu_score >= 0.7 else 0.0
        
        return quality_label, bleu_score
        
    except Exception as e:
        logger.warning(f"Failed to compute BLEU: {e}")
        return 0.0, 0.0


def process_single_example(
    example: Dict,
    models: Dict[str, ModelInference],
    reference_model: ModelInference,
    bleu_metric
) -> List[TrainingSample]:
    """Process single example through all models."""
    samples = []
    
    prompt = example["prompt"]
    
    # Generate reference output with largest model
    try:
        ref_output, ref_time, ref_meta = reference_model.generate(prompt)
    except Exception as e:
        logger.error(f"Failed to generate reference: {e}")
        return samples
    
    # Generate with each model and create training samples
    stage_map = {"7b": 0, "14b": 1, "32b": 2, "72b": 3}
    
    for model_size, model in models.items():
        if model_size == "72b":  # Skip reference model
            continue
            
        try:
            # Generate output
            output, gen_time, metadata = model.generate(prompt)
            
            # Extract features
            stage_id = stage_map[model_size]
            features = extract_features(prompt, output, metadata, stage_id)
            
            # Compute quality
            quality_label, bleu_score = compute_quality_score(output, ref_output, bleu_metric)
            
            # Create sample
            sample = TrainingSample(
                prompt=prompt,
                stage_id=stage_id,
                model_output=output,
                reference_output=ref_output,
                features=features,
                quality_score=quality_label,
                bleu_score=bleu_score,
                generation_time=gen_time,
                prompt_tokens=metadata["prompt_tokens"],
                completion_tokens=metadata["completion_tokens"]
            )
            
            samples.append(sample)
            
        except Exception as e:
            logger.error(f"Failed to process {model_size}: {e}")
            traceback.print_exc()
    
    return samples


def generate_training_data(
    dataset_path: Path,
    model_paths: Dict[str, str],
    output_dir: Path,
    max_samples: int = 1000,
    batch_size: int = 10
) -> Path:
    """Generate training data using multiple models."""
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    if len(dataset) > max_samples:
        dataset = dataset[:max_samples]
    
    # Initialize models
    models = {}
    device_allocation = {
        "7b": [0],
        "14b": [1],
        "32b": [2, 3],
        "72b": [4, 5, 6, 7]
    }
    
    for size, devices in device_allocation.items():
        model_path = model_paths.get(f"qwen3_{size}")
        if model_path and Path(model_path).exists():
            model = ModelInference(size, model_path, devices)
            model.load()
            models[size] = model
        else:
            logger.warning(f"Model path not found for {size}: {model_path}")
    
    if "72b" not in models:
        logger.error("72B model required as reference but not found!")
        return None
    
    reference_model = models["72b"]
    
    # Initialize BLEU metric
    bleu_metric = load("bleu")
    
    # Process examples
    all_samples = []
    
    logger.info(f"Processing {len(dataset)} examples...")
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating data"):
        batch = dataset[i:i+batch_size]
        
        for example in batch:
            samples = process_single_example(example, models, reference_model, bleu_metric)
            all_samples.extend(samples)
            
            # Save periodically
            if len(all_samples) % 100 == 0:
                logger.info(f"Generated {len(all_samples)} training samples")
    
    # Convert to serializable format
    training_data = []
    for sample in all_samples:
        data = asdict(sample)
        training_data.append(data)
    
    # Save training data
    output_file = output_dir / "training_data.json"
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Saved {len(training_data)} training samples to {output_file}")
    
    # Save feature statistics
    features = np.array([s.features for s in all_samples])
    feature_stats = {
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "min": features.min(axis=0).tolist(),
        "max": features.max(axis=0).tolist()
    }
    
    stats_file = output_dir / "feature_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset JSON file")
    parser.add_argument("--model-paths", type=str, required=True,
                       help="Path to model_paths.json")
    parser.add_argument("--output-dir", type=str,
                       default=f"/raid/{os.environ.get('USER', 'sasaki')}/training_data",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum samples to generate")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load model paths
    with open(args.model_paths, 'r') as f:
        model_paths = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    dataset_path = Path(args.dataset)
    
    output_file = generate_training_data(
        dataset_path=dataset_path,
        model_paths=model_paths,
        output_dir=output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    if output_file:
        logger.info(f"\nTraining data generation complete!")
        logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()