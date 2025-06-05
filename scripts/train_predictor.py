#!/usr/bin/env python3
"""
Train quality predictor for adaptive speculative decoding
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.predictor import QualityPredictor, FeatureExtractor, PredictorTrainer
from src.models.stage import Stage, StageManager, StageConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorDataset(Dataset):
    """Dataset for training quality predictor"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_training_data(
    stages: List[Stage],
    dataset_samples: List[Dict],
    num_samples: int,
    quality_threshold: float = 0.8
) -> List[Dict]:
    """
    Generate training data by running models on dataset samples
    
    Args:
        stages: List of model stages
        dataset_samples: List of prompts/references
        num_samples: Number of samples to generate
        quality_threshold: BLEU threshold for acceptance
        
    Returns:
        List of training samples
    """
    from evaluate import load
    bleu = load("bleu")
    
    training_data = []
    feature_extractor = FeatureExtractor()
    
    # Reference model (70B) for ground truth
    reference_stage = stages[-1]
    
    logger.info(f"Generating {num_samples} training samples...")
    
    for i, sample in enumerate(tqdm(dataset_samples[:num_samples])):
        prompt = sample["prompt"]
        
        # Generate reference output with 70B model
        try:
            ref_outputs, _, _ = reference_stage.generate(
                [prompt], max_tokens=128, temperature=0.7
            )
            reference_output = ref_outputs[0]
        except Exception as e:
            logger.warning(f"Failed to generate reference for sample {i}: {e}")
            continue
        
        # Test each draft stage
        for stage_idx, stage in enumerate(stages[:-1]):  # Exclude 70B
            try:
                # Generate draft output
                draft_outputs, draft_logprobs, _ = stage.generate(
                    [prompt], max_tokens=128, temperature=0.7
                )
                draft_output = draft_outputs[0]
                
                # Compute quality score
                try:
                    bleu_score = bleu.compute(
                        predictions=[draft_output],
                        references=[[reference_output]]
                    )["bleu"]
                except:
                    bleu_score = 0.0
                
                # Binary label based on threshold
                label = 1 if bleu_score >= quality_threshold else 0
                
                # Extract features
                features = feature_extractor.extract(
                    prompt=prompt,
                    draft_output=draft_output,
                    draft_logprobs=draft_logprobs[0] if draft_logprobs else None,
                    stage_id=stage_idx
                )
                
                training_data.append({
                    "features": features.tolist(),
                    "label": label,
                    "stage": stage_idx,
                    "bleu": bleu_score,
                    "prompt_length": len(prompt.split()),
                    "output_length": len(draft_output.split())
                })
                
            except Exception as e:
                logger.warning(f"Failed to process stage {stage_idx} for sample {i}: {e}")
                continue
    
    logger.info(f"Generated {len(training_data)} training samples")
    
    # Log class distribution
    labels = [sample["label"] for sample in training_data]
    pos_ratio = np.mean(labels)
    logger.info(f"Positive class ratio: {pos_ratio:.3f}")
    
    return training_data


def load_dataset_samples(config: Dict) -> List[Dict]:
    """Load samples from configured datasets"""
    from datasets import load_dataset
    
    samples = []
    
    for dataset_config in config["data_generation"]["datasets"]:
        dataset_name = dataset_config["name"]
        weight = dataset_config["weight"]
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            if dataset_name == "mmlu":
                dataset = load_dataset("cais/mmlu", "all", split="test")
                dataset_samples = [
                    {"prompt": f"Question: {sample['question']}\nAnswer:"}
                    for sample in dataset
                ]
                
            elif dataset_name == "humaneval":
                dataset = load_dataset("openai_humaneval", split="test")
                dataset_samples = [
                    {"prompt": sample["prompt"]}
                    for sample in dataset
                ]
                
            elif dataset_name == "hotpotqa":
                dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
                dataset_samples = [
                    {"prompt": f"Question: {sample['question']}\nAnswer:"}
                    for sample in dataset
                ]
                
            elif dataset_name == "alpacaeval":
                # Placeholder - would need actual AlpacaEval dataset
                dataset_samples = [
                    {"prompt": "Explain the concept of machine learning."},
                    {"prompt": "Write a short story about a robot."},
                    {"prompt": "What are the benefits of renewable energy?"}
                ] * 100  # Repeat for testing
                
            elif dataset_name == "longbench":
                # Placeholder - would need actual LongBench dataset
                dataset_samples = [
                    {"prompt": "Summarize the following long document: " + "Lorem ipsum " * 100}
                ] * 50
            
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            # Sample according to weight
            num_samples = int(len(dataset_samples) * weight)
            samples.extend(dataset_samples[:num_samples])
            
            logger.info(f"Added {num_samples} samples from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    logger.info(f"Total samples loaded: {len(samples)}")
    return samples


def train_predictor(
    training_data: List[Dict],
    config: Dict,
    save_path: str
) -> QualityPredictor:
    """
    Train the quality predictor model
    
    Args:
        training_data: List of training samples
        config: Training configuration
        save_path: Path to save the trained model
        
    Returns:
        Trained QualityPredictor model
    """
    # Extract features and labels
    features = np.array([sample["features"] for sample in training_data])
    labels = np.array([sample["label"] for sample in training_data])
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, 
        test_size=config["predictor"]["data"]["val_split"],
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Create datasets
    train_dataset = PredictorDataset(X_train, y_train)
    val_dataset = PredictorDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["predictor"]["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["predictor"]["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = QualityPredictor(
        feature_dim=config["predictor"]["model"]["feature_dim"],
        hidden_dims=[config["predictor"]["model"]["hidden_dim"]],
        dropout=config["predictor"]["model"]["dropout"]
    )
    
    # Initialize trainer
    trainer = PredictorTrainer(
        model=model,
        learning_rate=config["predictor"]["training"]["learning_rate"],
        weight_decay=config["predictor"]["training"]["weight_decay"]
    )
    
    # Class weights for imbalanced dataset
    pos_weight = len(y_train) / (2 * np.sum(y_train)) if np.sum(y_train) > 0 else 1.0
    class_weights = torch.tensor([1.0, pos_weight])
    
    if torch.cuda.is_available():
        model = model.cuda()
        class_weights = class_weights.cuda()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config["predictor"]["training"]["num_epochs"]):
        # Training
        model.train()
        train_losses = []
        train_accuracies = []
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if torch.cuda.is_available():
                batch_features = batch_features.cuda()
                batch_labels = batch_labels.cuda()
            
            # Compute class weights for this batch
            batch_weights = class_weights[batch_labels.long()] if config["predictor"]["data"]["balance_classes"] else None
            
            metrics = trainer.train_step(batch_features, batch_labels, batch_weights)
            train_losses.append(metrics["loss"])
            train_accuracies.append(metrics["accuracy"])
        
        # Validation
        model.eval()
        val_losses = []
        val_predictions = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                if torch.cuda.is_available():
                    batch_features = batch_features.cuda()
                    batch_labels = batch_labels.cuda()
                
                metrics = trainer.evaluate(batch_features, batch_labels)
                val_losses.append(metrics["loss"])
                
                # Collect predictions for metrics
                predictions = model(batch_features).squeeze(-1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())
        
        # Compute epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)
        avg_val_loss = np.mean(val_losses)
        
        val_predictions = np.array(val_predictions)
        val_labels_array = np.array(val_labels_list)
        
        val_acc = accuracy_score(val_labels_array, val_predictions > 0.5)
        val_auc = roc_auc_score(val_labels_array, val_predictions)
        
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Log to wandb if enabled
        if config["logging"]["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc
            })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= config["predictor"]["training"]["early_stopping_patience"]:
            logger.info("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train quality predictor")
    parser.add_argument("--config", default="configs/training.yaml", help="Training config file")
    parser.add_argument("--num-samples", type=int, help="Override number of training samples")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--data-file", help="Use existing training data file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with args
    if args.num_samples:
        config["data_generation"]["num_samples"] = args.num_samples
    
    if args.no_wandb:
        config["logging"]["use_wandb"] = False
    
    # Initialize wandb
    if config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["project_name"],
            config=config
        )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load or generate training data
    if args.data_file and os.path.exists(args.data_file):
        logger.info(f"Loading existing training data from {args.data_file}")
        with open(args.data_file, 'r') as f:
            training_data = json.load(f)
    else:
        # Initialize models for data generation
        logger.info("Initializing models for data generation...")
        
        # Note: This is a simplified version - in practice you'd load from checkpoints
        stage_configs = [
            StageConfig("meta-llama/Llama-3.2-8B", "8b", 1, 0.8, True, 1.0),
            StageConfig("meta-llama/Llama-3.1-13B", "13b", 2, 0.8, True, 1.6),
            StageConfig("codellama/CodeLlama-34b-hf", "34b", 2, 0.8, True, 4.2),
            StageConfig("meta-llama/Llama-3.1-70B", "70b", 4, 0.9, True, 8.8)
        ]
        
        gpu_allocation = {
            "8b": [0], "13b": [1, 2], "34b": [3, 4], "70b": [5, 6, 7]
        }
        
        stage_manager = StageManager(stage_configs, gpu_allocation)
        stages = [stage_manager.get_stage(size) for size in ["8b", "13b", "34b", "70b"]]
        
        # Load dataset samples
        dataset_samples = load_dataset_samples(config)
        
        # Generate training data
        training_data = generate_training_data(
            stages=stages,
            dataset_samples=dataset_samples,
            num_samples=config["data_generation"]["num_samples"],
            quality_threshold=config["data_generation"]["quality_threshold"]["bleu"]
        )
        
        # Save training data
        data_file = f"{args.output_dir}/training_data.json"
        with open(data_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Training data saved to {data_file}")
    
    # Train predictor
    logger.info("Starting predictor training...")
    save_path = f"{args.output_dir}/predictor.pt"
    
    model = train_predictor(
        training_data=training_data,
        config=config,
        save_path=save_path
    )
    
    logger.info(f"Training completed. Model saved to {save_path}")
    
    # Test inference speed
    logger.info("Testing inference speed...")
    feature_extractor = FeatureExtractor()
    
    dummy_features = np.random.randn(256)
    start_time = time.time()
    
    for _ in range(1000):
        with torch.no_grad():
            features_tensor = torch.tensor(dummy_features, dtype=torch.float32)
            if torch.cuda.is_available():
                features_tensor = features_tensor.cuda()
            prob = model(features_tensor.unsqueeze(0)).item()
    
    avg_time_ms = (time.time() - start_time) * 1000 / 1000
    logger.info(f"Average inference time: {avg_time_ms:.3f}ms")
    
    if avg_time_ms > 0.3:
        logger.warning("Inference time exceeds 0.3ms target!")
    else:
        logger.info("✓ Inference time meets target")


if __name__ == "__main__":
    import time
    main()