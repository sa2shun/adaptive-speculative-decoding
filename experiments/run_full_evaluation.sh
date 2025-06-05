#!/bin/bash
"""
Full evaluation script for adaptive speculative decoding
"""

set -e  # Exit on error

echo "Starting full evaluation of Adaptive Speculative Decoding..."

# Configuration
RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
LAMBDA_VALUES="0.1 0.2 0.5 1.0 2.0 5.0 10.0"
DATASETS="mmlu humaneval hotpotqa alpacaeval longbench"
NUM_SAMPLES=1000
CONFIG_DIR="configs"

# Create results directory
mkdir -p ${RESULTS_DIR}
echo "Results will be saved to: ${RESULTS_DIR}"

# Log system info
echo "=== System Information ===" > ${RESULTS_DIR}/system_info.txt
nvidia-smi >> ${RESULTS_DIR}/system_info.txt
echo "" >> ${RESULTS_DIR}/system_info.txt
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" >> ${RESULTS_DIR}/system_info.txt

# Start server in background
echo "Starting adaptive decoding server..."
python -m src.serving.server --config ${CONFIG_DIR}/serving.yaml &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 30

# Check if server is running
if ! ps -p ${SERVER_PID} > /dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

echo "Server started successfully (PID: ${SERVER_PID})"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if ps -p ${SERVER_PID} > /dev/null; then
        kill ${SERVER_PID}
        echo "Server stopped"
    fi
}
trap cleanup EXIT

# Wait for server to be ready
echo "Checking server health..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "Server is ready!"
        break
    fi
    echo "Waiting for server... (attempt $i/10)"
    sleep 5
done

# Verify server is responding
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "ERROR: Server not responding"
    exit 1
fi

# Main evaluation loop
echo "Starting evaluation runs..."

total_runs=$(($(echo ${LAMBDA_VALUES} | wc -w) * $(echo ${DATASETS} | wc -w)))
current_run=0

for lambda in ${LAMBDA_VALUES}; do
    for dataset in ${DATASETS}; do
        current_run=$((current_run + 1))
        echo ""
        echo "=== Run ${current_run}/${total_runs}: Lambda=${lambda}, Dataset=${dataset} ==="
        
        # Update lambda parameter
        echo "Setting lambda to ${lambda}..."
        curl -s -X POST "http://localhost:8000/update_lambda" \
            -H "Content-Type: application/json" \
            -d "{\"lambda_value\": ${lambda}}" || {
            echo "ERROR: Failed to update lambda"
            continue
        }
        
        # Reset statistics
        curl -s -X POST "http://localhost:8000/reset_stats" || {
            echo "WARNING: Failed to reset stats"
        }
        
        # Run evaluation
        output_file="${RESULTS_DIR}/lambda_${lambda}_${dataset}.json"
        
        echo "Running evaluation..."
        python experiments/evaluate_pipeline.py \
            --lambda ${lambda} \
            --datasets ${dataset} \
            --num-samples ${NUM_SAMPLES} \
            --output-file ${output_file} \
            --server-url "http://localhost:8000" || {
            echo "ERROR: Evaluation failed for lambda=${lambda}, dataset=${dataset}"
            continue
        }
        
        echo "Results saved to ${output_file}"
        
        # Get server stats
        stats_file="${RESULTS_DIR}/stats_lambda_${lambda}_${dataset}.json"
        curl -s "http://localhost:8000/stats" > ${stats_file}
        
        # Brief cooldown
        sleep 10
    done
done

echo ""
echo "=== All evaluations completed ==="

# Aggregate results
echo "Aggregating results..."
python experiments/aggregate_results.py \
    --results-dir ${RESULTS_DIR} \
    --output-file ${RESULTS_DIR}/aggregated_results.json

# Generate plots
echo "Generating plots..."
python experiments/plot_results.py \
    --results-file ${RESULTS_DIR}/aggregated_results.json \
    --output-dir ${RESULTS_DIR}/plots

# Create summary report
echo "Creating summary report..."
python experiments/create_report.py \
    --results-file ${RESULTS_DIR}/aggregated_results.json \
    --output-file ${RESULTS_DIR}/summary_report.md

echo ""
echo "=== Evaluation Summary ==="
echo "Results directory: ${RESULTS_DIR}"
echo "Total runs: ${total_runs}"
echo "Lambda values tested: ${LAMBDA_VALUES}"
echo "Datasets tested: ${DATASETS}"
echo "Samples per dataset: ${NUM_SAMPLES}"
echo ""
echo "Key files:"
echo "  - Aggregated results: ${RESULTS_DIR}/aggregated_results.json"
echo "  - Summary report: ${RESULTS_DIR}/summary_report.md"
echo "  - Plots: ${RESULTS_DIR}/plots/"
echo ""
echo "Evaluation completed successfully!"

# Display quick results summary
if [ -f "${RESULTS_DIR}/summary_report.md" ]; then
    echo ""
    echo "=== Quick Summary ==="
    head -20 ${RESULTS_DIR}/summary_report.md
fi