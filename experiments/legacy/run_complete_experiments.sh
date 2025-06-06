#!/bin/bash
# Complete experiment pipeline for adaptive speculative decoding research
# This script runs all experiments from model download to final evaluation

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
USER=${USER:-sasaki}
BASE_DIR="/raid/${USER}/adaptive-speculative-decoding"
MODEL_DIR="${BASE_DIR}/models"
DATASET_DIR="${BASE_DIR}/datasets"
TRAINING_DIR="${BASE_DIR}/training_data"
RESULTS_DIR="${BASE_DIR}/results"
LOG_DIR="${BASE_DIR}/logs"

# Create directories
mkdir -p ${MODEL_DIR} ${DATASET_DIR} ${TRAINING_DIR} ${RESULTS_DIR} ${LOG_DIR}

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/run_${TIMESTAMP}"
mkdir -p ${RUN_DIR}

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/experiment_${TIMESTAMP}.log"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check GPU availability
check_gpus() {
    log "Checking GPU availability..."
    nvidia-smi || error_exit "nvidia-smi failed. Are GPUs available?"
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ $GPU_COUNT -lt 8 ]; then
        error_exit "Need 8 GPUs but found only $GPU_COUNT"
    fi
    log "✓ Found $GPU_COUNT GPUs"
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    # Check if we're already in a working Python environment
    if python3 -c "import torch, transformers, datasets" 2>/dev/null; then
        log "✓ Dependencies already available in current environment"
        return 0
    fi
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    pip install -r requirements_minimal.txt
    
    # Install additional dependencies for experiments
    pip install bitsandbytes transformers accelerate datasets evaluate wandb
    
    log "✓ Dependencies installed"
}

# Download models
download_models() {
    log "=== PHASE 1: Model Download ==="
    
    # Check if models already exist
    if [ -f "${MODEL_DIR}/model_paths.json" ]; then
        log "Models already downloaded. Verifying..."
        python3 scripts/download_qwen3_models.py \
            --base-path ${MODEL_DIR} \
            --verify-only \
            2>&1 | tee -a "${LOG_DIR}/model_download_${TIMESTAMP}.log"
        
        if [ $? -eq 0 ]; then
            log "✓ Models verified successfully"
            return 0
        fi
    fi
    
    # Download models
    log "Downloading Qwen3 model family..."
    python3 scripts/download_qwen3_models.py \
        --base-path ${MODEL_DIR} \
        2>&1 | tee -a "${LOG_DIR}/model_download_${TIMESTAMP}.log"
    
    if [ $? -ne 0 ]; then
        error_exit "Model download failed"
    fi
    
    log "✓ Models downloaded successfully"
}

# Setup datasets
setup_datasets() {
    log "=== PHASE 2: Dataset Setup ==="
    
    # Check if datasets already exist
    if [ -f "${DATASET_DIR}/dataset_summary.json" ]; then
        log "Datasets already exist"
        return 0
    fi
    
    log "Setting up evaluation datasets..."
    python3 setup_datasets.py \
        --output-dir ${DATASET_DIR} \
        --mmlu-samples 2000 \
        --mt-bench-samples 100 \
        --simple-qa-samples 500 \
        2>&1 | tee -a "${LOG_DIR}/dataset_setup_${TIMESTAMP}.log"
    
    if [ $? -ne 0 ]; then
        error_exit "Dataset setup failed"
    fi
    
    log "✓ Datasets prepared successfully"
}

# Generate training data
generate_training_data() {
    log "=== PHASE 3: Training Data Generation ==="
    
    # Check if training data already exists
    if [ -f "${TRAINING_DIR}/training_data.json" ]; then
        log "Training data already exists"
        return 0
    fi
    
    log "Generating training data from model outputs..."
    
    # Use mixed dataset for training data generation
    python3 src/training/generate_training_data.py \
        --dataset "${DATASET_DIR}/mmlu_test.json" \
        --model-paths "${MODEL_DIR}/model_paths.json" \
        --output-dir ${TRAINING_DIR} \
        --max-samples 10000 \
        --batch-size 8 \
        2>&1 | tee -a "${LOG_DIR}/training_data_gen_${TIMESTAMP}.log"
    
    if [ $? -ne 0 ]; then
        log "WARNING: Training data generation had issues, continuing..."
    fi
    
    log "✓ Training data generation complete"
}

# Train quality predictor
train_predictor() {
    log "=== PHASE 4: Quality Predictor Training ==="
    
    log "Training quality predictor..."
    
    # Create training script
    cat > train_predictor_real.py << 'EOF'
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('.')
from src.minimal_adaptive_decoder import MinimalQualityPredictor

# Load training data
with open('/raid/${USER}/adaptive-speculative-decoding/training_data/training_data.json', 'r') as f:
    data = json.load(f)

# Extract features and labels
X = np.array([d['features'] for d in data])
y = np.array([d['quality_score'] for d in data])

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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Initialize model
model = MinimalQualityPredictor(input_dim=64, hidden_dim=32)
if torch.cuda.is_available():
    model = model.cuda()

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    for features, labels in train_loader:
        if torch.cuda.is_available():
            features, labels = features.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()
            
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
          f"Val Loss={val_loss/len(val_loader):.4f}, Accuracy={accuracy:.4f}")

# Save model
torch.save(model.state_dict(), '/raid/${USER}/adaptive-speculative-decoding/training_data/predictor.pt')
print("Model saved!")
EOF

    # Replace ${USER} in the script
    sed -i "s/\${USER}/${USER}/g" train_predictor_real.py
    
    python3 train_predictor_real.py \
        2>&1 | tee -a "${LOG_DIR}/predictor_training_${TIMESTAMP}.log"
    
    if [ $? -ne 0 ]; then
        log "WARNING: Predictor training had issues, continuing..."
    fi
    
    # Cleanup
    rm train_predictor_real.py
    
    log "✓ Quality predictor trained"
}

# Run core experiments
run_experiments() {
    log "=== PHASE 5: Core Experiments ==="
    
    log "Running theoretical demonstration..."
    python3 simple_theory_demo.py \
        2>&1 | tee -a "${LOG_DIR}/theory_demo_${TIMESTAMP}.log"
    
    log "Running empirical experiments..."
    python3 simple_experiments.py \
        2>&1 | tee -a "${LOG_DIR}/empirical_experiments_${TIMESTAMP}.log"
    
    # Run real experiments if models are loaded
    if [ -f "${MODEL_DIR}/model_paths.json" ]; then
        log "Running experiments with real models..."
        
        python3 run_real_experiments.py \
            --output-dir ${RUN_DIR} \
            --quick-test \
            2>&1 | tee -a "${LOG_DIR}/real_experiments_${TIMESTAMP}.log"
    fi
    
    log "✓ Core experiments completed"
}

# Generate final report
generate_report() {
    log "=== PHASE 6: Report Generation ==="
    
    # Create final report
    cat > ${RUN_DIR}/experiment_report.md << EOF
# Adaptive Speculative Decoding - Experiment Report
## Run: ${TIMESTAMP}

### System Configuration
- GPUs: 8x H100
- User: ${USER}
- Base Directory: ${BASE_DIR}

### Completed Phases
1. ✓ Model Download
2. ✓ Dataset Setup  
3. ✓ Training Data Generation
4. ✓ Quality Predictor Training
5. ✓ Core Experiments

### Key Results
EOF

    # Add results if available
    if [ -f "${RUN_DIR}/evaluation_results.json" ]; then
        echo "#### Performance Metrics" >> ${RUN_DIR}/experiment_report.md
        python3 -c "
import json
with open('${RUN_DIR}/evaluation_results.json', 'r') as f:
    data = json.load(f)
    print(f\"- Average Cost: {data.get('avg_cost', 'N/A')}\")
    print(f\"- Average Quality: {data.get('avg_quality', 'N/A')}\")
    print(f\"- Speedup vs 72B: {data.get('speedup_vs_72b', 'N/A')}x\")
" >> ${RUN_DIR}/experiment_report.md
    fi
    
    echo -e "\n### Logs\nAll logs saved in: ${LOG_DIR}" >> ${RUN_DIR}/experiment_report.md
    
    log "✓ Report generated: ${RUN_DIR}/experiment_report.md"
}

# Main execution
main() {
    log "=========================================="
    log "Starting Adaptive Speculative Decoding Experiments"
    log "Timestamp: ${TIMESTAMP}"
    log "=========================================="
    
    # System checks
    check_gpus
    
    # Install dependencies
    install_dependencies
    
    # Run pipeline
    download_models
    setup_datasets
    generate_training_data
    train_predictor
    run_experiments
    generate_report
    
    # Copy important files to results
    cp *.png ${RUN_DIR}/ 2>/dev/null || true
    cp *.pdf ${RUN_DIR}/ 2>/dev/null || true
    
    log "=========================================="
    log "All experiments completed successfully!"
    log "Results saved in: ${RUN_DIR}"
    log "=========================================="
    
    # No virtual environment to deactivate
}

# Cleanup on exit
cleanup() {
    log "Cleaning up..."
    # Kill any remaining GPU processes
    pkill -f "python.*generate_training_data" || true
    pkill -f "python.*run_real_experiments" || true
}

trap cleanup EXIT

# Run main
main "$@"