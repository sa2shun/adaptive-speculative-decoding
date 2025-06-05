#!/bin/bash
# Environment setup script for Adaptive Speculative Decoding

set -e  # Exit on error

echo "Setting up Adaptive Speculative Decoding environment..."

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU support may not be available."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

# Create conda environment
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda create -n adaptive-sd python=3.10 -y
    echo "Conda environment created. Activate with: conda activate adaptive-sd"
else
    echo "Conda not found. Using system Python..."
fi

# Create directory structure
echo "Creating project directories..."
mkdir -p src/{models,algorithms,serving,utils}
mkdir -p configs
mkdir -p scripts
mkdir -p experiments/{ablation,baselines}
mkdir -p tests
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Create __init__.py files
touch src/__init__.py
touch src/models/__init__.py
touch src/algorithms/__init__.py
touch src/serving/__init__.py
touch src/utils/__init__.py

# Download required NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate conda environment: conda activate adaptive-sd"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Download models: python scripts/download_models.py"
echo "4. Train predictor: python scripts/train_predictor.py"