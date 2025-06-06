#!/usr/bin/env python3
"""
Train large-scale quality predictor for adaptive speculative decoding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityPredictorDataset(Dataset):
    """Dataset for quality predictor training"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AdvancedQualityPredictor(nn.Module):
    """Advanced neural network for quality prediction"""
    
    def __init__(
        self, 
        input_dim: int = 8,
        hidden_dims: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32, 16]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Probability output [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def load_training_data(data_dir: str) -> tuple:
    """Load training data from saved files"""
    
    data_path = Path(data_dir)
    
    # Load features and labels
    features = np.load(data_path / "features.npy")
    labels = np.load(data_path / "labels.npy")
    
    # Load metadata
    with open(data_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
    
    return features, labels, metadata


def create_data_loaders(
    features: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> tuple:
    """Create train, validation, and test data loaders"""
    
    dataset = QualityPredictorDataset(features, labels)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: str = "cuda"
) -> dict:
    """Train the quality predictor model"""
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    logger.info(f"Starting training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - labels)).item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - labels)).item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mae /= len(train_loader)
        val_mae /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(current_lr)
        
        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return history


def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: str = "cuda"
) -> dict:
    """Evaluate the trained model"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    logger.info(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return metrics


def plot_training_history(history: dict, save_path: str = None):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[0, 1].plot(history['train_mae'], label='Train MAE', alpha=0.8)
    axes[0, 1].plot(history['val_mae'], label='Validation MAE', alpha=0.8)
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'], alpha=0.8)
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff, alpha=0.8)
    axes[1, 1].set_title('Overfitting Indicator (Val Loss - Train Loss)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved: {save_path}")
        plt.close()


def plot_predictions(metrics: dict, save_path: str = None):
    """Plot prediction results"""
    
    predictions = metrics['predictions']
    labels = metrics['labels']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    axes[0].scatter(labels, predictions, alpha=0.5, s=1)
    axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title(f'Predictions vs True Values (R² = {metrics["r2"]:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = predictions - labels
    axes[1].scatter(predictions, residuals, alpha=0.5, s=1)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('Predictions')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction plot saved: {save_path}")
        plt.close()


def save_model(model: nn.Module, history: dict, metrics: dict, output_dir: str):
    """Save trained model and results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / "quality_predictor.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'hidden_dims': model.hidden_dims,
            'dropout_rate': model.dropout_rate,
            'use_batch_norm': model.use_batch_norm
        },
        'training_history': history,
        'test_metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'labels']}
    }, model_path)
    
    logger.info(f"Model saved: {model_path}")
    
    # Save detailed results
    results = {
        'final_metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'labels']},
        'training_history': history
    }
    
    with open(output_path / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train quality predictor")
    parser.add_argument("--data-dir", default="/raid/sasaki/adaptive-sd-training-data",
                       help="Directory containing training data")
    parser.add_argument("--output-dir", default="/raid/sasaki/adaptive-sd-models/quality-predictor",
                       help="Output directory for trained model")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dims", nargs='+', type=int, default=[128, 64, 32, 16],
                       help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    print("🚀 Large-Scale Quality Predictor Training")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden dims: {args.hidden_dims}")
    
    # Load data
    features, labels, metadata = load_training_data(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        features, labels, batch_size=args.batch_size
    )
    
    # Create model
    model = AdvancedQualityPredictor(
        input_dim=metadata['feature_dimension'],
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout
    )
    
    logger.info(f"Model architecture: {model}")
    
    # Train model
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device=args.device)
    
    # Plot results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_training_history(history, save_path=str(output_path / "training_history.png"))
    plot_predictions(metrics, save_path=str(output_path / "predictions.png"))
    
    # Save model
    save_model(model, history, metrics, args.output_dir)
    
    print(f"\n✅ Training completed!")
    print(f"   Final Test R²: {metrics['r2']:.4f}")
    print(f"   Final Test MAE: {metrics['mae']:.4f}")
    print(f"   Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()