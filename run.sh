#!/bin/bash
"""
Quick start script for Adaptive Speculative Decoding
"""

set -e

echo "🚀 Adaptive Speculative Decoding - Quick Start"
echo "=============================================="

# Check if we're in Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running in Docker container"
    IN_DOCKER=true
else
    echo "⚠️  Running on host system"
    IN_DOCKER=false
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Set up environment and download models"
    echo "  train     - Train quality predictor"
    echo "  serve     - Start API server"
    echo "  demo      - Run quick demo"
    echo "  eval      - Run full evaluation"
    echo "  docker    - Build and run with Docker"
    echo "  help      - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Initial setup"
    echo "  $0 serve     # Start server on port 8000"
    echo "  $0 demo      # Test with demo requests"
}

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    "setup")
        echo "🔧 Setting up environment..."
        
        # Setup environment
        if [ "$IN_DOCKER" = false ]; then
            bash scripts/setup_env.sh
            echo "📦 Installing dependencies..."
            pip install -r requirements.txt
        fi
        
        # Download models (smaller models first)
        echo "📥 Downloading models..."
        echo "⚠️  This will download ~250GB of models. Continue? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            python scripts/download_models.py --models "8b,13b,34b,70b"
        else
            echo "Skipping model download. You can run it later with:"
            echo "  python scripts/download_models.py --models 8b,13b,34b,70b"
        fi
        
        echo "✅ Setup completed!"
        ;;
        
    "train")
        echo "🧠 Training quality predictor..."
        
        # Check if models exist
        if [ ! -d "checkpoints/8b" ]; then
            echo "❌ Models not found. Run 'setup' first."
            exit 1
        fi
        
        python scripts/train_predictor.py \
            --config configs/training.yaml \
            --num-samples 10000 \
            --output-dir checkpoints
        
        echo "✅ Training completed!"
        ;;
        
    "serve")
        echo "🖥️  Starting API server..."
        
        # Check if predictor exists
        if [ ! -f "checkpoints/predictor.pt" ]; then
            echo "⚠️  Predictor not found. Starting with random weights."
            echo "   Run 'train' command to train the predictor first."
        fi
        
        # Start server
        python -m src.serving.server \
            --config configs/serving.yaml \
            --host 0.0.0.0 \
            --port 8000
        ;;
        
    "demo")
        echo "🎯 Running demo..."
        
        # Check if server is running
        if ! curl -s http://localhost:8000/health > /dev/null; then
            echo "❌ Server not running. Start with 'serve' command first."
            exit 1
        fi
        
        python examples/quick_demo.py
        ;;
        
    "eval")
        echo "📊 Running full evaluation..."
        
        # Check if server is running
        if ! curl -s http://localhost:8000/health > /dev/null; then
            echo "❌ Server not running. Start with 'serve' command first."
            exit 1
        fi
        
        bash experiments/run_full_evaluation.sh
        ;;
        
    "docker")
        echo "🐳 Building and running with Docker..."
        
        # Build Docker image
        echo "Building Docker image..."
        docker build -t adaptive-sd:latest .
        
        # Run with docker-compose
        echo "Starting with Docker Compose..."
        docker-compose up -d
        
        echo "✅ Docker container started!"
        echo "   Server: http://localhost:8000"
        echo "   Notebook: http://localhost:8888"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
        ;;
        
    "help"|*)
        show_usage
        ;;
esac