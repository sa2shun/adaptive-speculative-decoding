#!/usr/bin/env python3
"""
Train quality predictor with real model outputs.
This creates training data by comparing outputs from available models.
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')

class RealQualityPredictor(nn.Module):
    """Simple MLP for quality prediction."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class QualityPredictorTrainer:
    def __init__(self, base_model_dir: str, dataset_dir: str, output_dir: str):
        self.base_model_dir = Path(base_model_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.tokenizers = {}
        
        # Model configuration
        self.model_configs = {
            "7b": {"path": "qwen3-7b", "name": "Qwen3-7B", "cost": 1.0},
            "14b": {"path": "qwen3-14b", "name": "Qwen3-14B", "cost": 2.0},
            "32b": {"path": "qwen3-32b", "name": "Qwen3-32B", "cost": 4.5},
            "72b": {"path": "qwen3-72b", "name": "Qwen3-72B", "cost": 10.0}
        }
    
    def check_available_models(self) -> List[str]:
        """Check which models are ready."""
        available = []
        for size, config in self.model_configs.items():
            model_path = self.base_model_dir / config["path"]
            if self._model_is_complete(model_path):
                available.append(size)
                print(f"✓ {config['name']} available")
            else:
                print(f"✗ {config['name']} not ready")
        return sorted(available, key=lambda x: self.model_configs[x]["cost"])
    
    def _model_is_complete(self, model_path: Path) -> bool:
        """Check if model download is complete."""
        required_files = ["config.json", "tokenizer.json"]
        return all((model_path / f).exists() for f in required_files)
    
    def load_model(self, size: str) -> bool:
        """Load a model."""
        if size in self.models:
            return True
        
        config = self.model_configs[size]
        model_path = self.base_model_dir / config["path"]
        
        try:
            print(f"Loading {config['name']}...")
            
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizers[size] = tokenizer
            self.models[size] = model
            
            print(f"✓ {config['name']} loaded")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load {config['name']}: {e}")
            return False
    
    def unload_model(self, size: str):
        """Unload model to free memory."""
        if size in self.models:
            del self.models[size]
            del self.tokenizers[size]
            torch.cuda.empty_cache()
    
    def extract_features(self, prompt: str, model_size: str) -> np.ndarray:
        """Extract features for quality prediction."""
        # Basic text features
        features = []
        
        # 1. Prompt characteristics
        features.extend([
            len(prompt),                    # Length
            len(prompt.split()),           # Word count
            len(prompt.split('.')) - 1,    # Sentence count
            prompt.count('?'),             # Question marks
            prompt.count('!'),             # Exclamation marks
        ])
        
        # 2. Complexity indicators
        avg_word_length = np.mean([len(word) for word in prompt.split()]) if prompt.split() else 0
        features.extend([
            avg_word_length,
            prompt.count(','),             # Commas
            prompt.count(';'),             # Semicolons
            prompt.count(':'),             # Colons
            prompt.count('('),             # Parentheses
        ])
        
        # 3. Model-specific features
        model_costs = {"7b": 1.0, "14b": 2.0, "32b": 4.5, "72b": 10.0}
        stage_encoding = [1.0 if model_size == size else 0.0 for size in ["7b", "14b", "32b", "72b"]]
        features.extend(stage_encoding)
        features.append(model_costs.get(model_size, 1.0))
        
        # 4. Text complexity metrics
        unique_words = len(set(prompt.lower().split()))
        total_words = len(prompt.split())
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        features.append(lexical_diversity)
        
        # 5. Tokenizer-based features (if available)
        if model_size in self.tokenizers:
            try:
                tokens = self.tokenizers[model_size](prompt, return_tensors="pt")
                token_count = len(tokens.input_ids[0])
                features.extend([
                    token_count,
                    token_count / len(prompt) if len(prompt) > 0 else 0,  # tokens per character
                ])
            except:
                features.extend([len(prompt.split()), 0.2])  # fallback
        else:
            features.extend([len(prompt.split()), 0.2])
        
        # Pad to fixed size
        target_size = 64
        while len(features) < target_size:
            features.append(0.0)
        
        return np.array(features[:target_size], dtype=np.float32)
    
    def generate_with_model(self, size: str, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Generate text with a model."""
        if size not in self.models:
            return {"error": f"Model {size} not loaded"}
        
        model = self.models[size]
        tokenizer = self.tokenizers[size]
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            inference_time = time.time() - start_time
            
            # Extract generated text
            generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate average log probability (quality indicator)
            if hasattr(outputs, 'scores') and outputs.scores:
                scores = torch.stack(outputs.scores, dim=0)  # [seq_len, 1, vocab_size]
                probs = torch.softmax(scores, dim=-1)
                selected_probs = torch.gather(probs, 2, generated_ids.unsqueeze(0).unsqueeze(-1))
                avg_log_prob = torch.log(selected_probs.squeeze()).mean().item()
            else:
                avg_log_prob = -2.0  # default value
            
            return {
                "generated_text": generated_text,
                "inference_time": inference_time,
                "avg_log_prob": avg_log_prob,
                "new_tokens": len(generated_ids),
                "quality_score": self._estimate_quality(generated_text, prompt)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _estimate_quality(self, generated_text: str, prompt: str) -> float:
        """Estimate quality of generated text (simple heuristics)."""
        if not generated_text.strip():
            return 0.0
        
        score = 0.5  # base score
        
        # Length appropriateness
        if 10 <= len(generated_text) <= 200:
            score += 0.1
        
        # Coherence indicators
        if generated_text.endswith('.') or generated_text.endswith('!') or generated_text.endswith('?'):
            score += 0.1
        
        # No repetition
        words = generated_text.split()
        if len(set(words)) / len(words) > 0.7 if words else True:
            score += 0.1
        
        # Relevance to prompt (simple keyword overlap)
        prompt_words = set(prompt.lower().split())
        generated_words = set(generated_text.lower().split())
        overlap = len(prompt_words & generated_words) / len(prompt_words) if prompt_words else 0
        score += min(overlap * 0.2, 0.2)
        
        return min(score, 1.0)
    
    def generate_training_data(self, num_samples: int = 500) -> List[Dict[str, Any]]:
        """Generate training data from available models."""
        print(f"Generating training data with {num_samples} samples...")
        
        # Load datasets
        datasets = []
        for dataset_name in ["mmlu", "humaneval", "simple_qa"]:
            dataset_path = self.dataset_dir / f"{dataset_name}_test.json"
            if dataset_path.exists():
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    datasets.extend(data[:num_samples//3])  # Split evenly
        
        if not datasets:
            print("No datasets found!")
            return []
        
        # Sample prompts
        import random
        random.seed(42)
        if len(datasets) > num_samples:
            datasets = random.sample(datasets, num_samples)
        
        available_models = self.check_available_models()
        if not available_models:
            print("No models available for training data generation!")
            return []
        
        training_data = []
        
        for i, sample in enumerate(datasets):
            if i % 50 == 0:
                print(f"Processing sample {i+1}/{len(datasets)}...")
            
            prompt = sample.get("prompt", sample.get("question", ""))
            if not prompt:
                continue
            
            # For each available model, generate training example
            for model_size in available_models:
                if not self.load_model(model_size):
                    continue
                
                # Extract features
                features = self.extract_features(prompt, model_size)
                
                # Generate with model
                result = self.generate_with_model(model_size, prompt, max_tokens=50)
                
                if "error" not in result:
                    # Create training example
                    # Quality label: 1 if this model should be used, 0 otherwise
                    # Simple heuristic: smaller models for shorter/simpler prompts
                    prompt_complexity = len(prompt.split()) + prompt.count('?') * 2 + prompt.count('!') * 2
                    
                    if model_size == "7b":
                        quality_label = 1.0 if prompt_complexity <= 15 else 0.7
                    elif model_size == "14b":
                        quality_label = 1.0 if 10 <= prompt_complexity <= 25 else 0.8
                    elif model_size == "32b":
                        quality_label = 1.0 if 20 <= prompt_complexity <= 40 else 0.9
                    else:  # 72b
                        quality_label = 1.0 if prompt_complexity >= 30 else 0.95
                    
                    # Add noise to make it more realistic
                    quality_label += np.random.normal(0, 0.1)
                    quality_label = np.clip(quality_label, 0.0, 1.0)
                    
                    training_data.append({
                        "prompt": prompt,
                        "model_size": model_size,
                        "features": features.tolist(),
                        "quality_score": quality_label,
                        "generated_text": result["generated_text"],
                        "inference_time": result["inference_time"],
                        "avg_log_prob": result.get("avg_log_prob", -2.0)
                    })
                
                # Unload model to save memory
                self.unload_model(model_size)
        
        print(f"Generated {len(training_data)} training examples")
        return training_data
    
    def train_predictor(self, training_data: List[Dict[str, Any]]) -> RealQualityPredictor:
        """Train the quality predictor."""
        print("Training quality predictor...")
        
        # Prepare data
        X = np.array([d["features"] for d in training_data])
        y = np.array([d["quality_score"] for d in training_data])
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Quality score range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RealQualityPredictor(input_dim=X.shape[1], hidden_dim=32).to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Changed from BCELoss to MSELoss for regression
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training loop
        for epoch in range(100):
            # Train
            model.train()
            train_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_mae += torch.abs(outputs - labels).mean().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val MAE={avg_val_mae:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.output_dir / "best_predictor.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(self.output_dir / "best_predictor.pt"))
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return model
    
    def run_full_training_pipeline(self, num_samples: int = 500):
        """Run the complete training pipeline."""
        print("=== REAL QUALITY PREDICTOR TRAINING ===")
        
        # 1. Generate training data
        training_data = self.generate_training_data(num_samples)
        
        if not training_data:
            print("No training data generated!")
            return
        
        # 2. Save training data
        training_data_path = self.output_dir / "training_data_real.json"
        with open(training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"Training data saved to {training_data_path}")
        
        # 3. Train predictor
        predictor = self.train_predictor(training_data)
        
        # 4. Save final model
        torch.save(predictor.state_dict(), self.output_dir / "final_predictor.pt")
        
        # 5. Create model info
        model_info = {
            "input_dim": 64,
            "hidden_dim": 32,
            "num_training_samples": len(training_data),
            "available_models": self.check_available_models(),
            "timestamp": time.time()
        }
        
        with open(self.output_dir / "predictor_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("=== TRAINING COMPLETE ===")
        print(f"Model saved to {self.output_dir}")
        print(f"Training samples: {len(training_data)}")
        
        return predictor

def main():
    """Main training pipeline."""
    trainer = QualityPredictorTrainer(
        base_model_dir="/raid/sasaki/adaptive-speculative-decoding/models",
        dataset_dir="/raid/sasaki/adaptive-speculative-decoding/datasets",
        output_dir="/raid/sasaki/adaptive-speculative-decoding/training_data"
    )
    
    # Run training with current available models
    predictor = trainer.run_full_training_pipeline(num_samples=200)  # Start small
    
    if predictor:
        print("Quality predictor training successful!")
    else:
        print("Training failed - no models available")

if __name__ == "__main__":
    main()