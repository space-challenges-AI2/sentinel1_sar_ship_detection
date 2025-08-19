#!/bin/bash

# Build and Run Script for SAR Ship Detection Pipeline
# This script builds the Docker image and runs the container

set -e

echo "Building SAR Ship Detection Pipeline Docker Image..."

# Get the project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Project root: $PROJECT_ROOT"
echo "Building from: $PROJECT_ROOT"

# Build the image from the project root directory
cd "$PROJECT_ROOT"
docker build -f utils/docker/Dockerfile-laptop -t sar-ship-detection:latest .

echo "Image built successfully!"

echo "Running container..."

# Check if NVIDIA runtime is available and working
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, running with GPU support..."
    # Run the container with GPU support
    docker run -it --rm \
        --gpus all \
        --ipc=host \
        --name sar-ship-detection \
        -v "$(pwd)/data:/workspace/data:ro" \
        -v "$(pwd)/weights:/workspace/weights:ro" \
        -v "$(pwd)/configs:/workspace/configs:ro" \
        -v "$(pwd)/test_ingest:/workspace/test_ingest" \
        -v "$(pwd)/test_work:/workspace/test_work" \
        -v "$(pwd)/test_detections:/workspace/test_detections" \
        -v "$(pwd)/test_thumbs:/workspace/test_thumbs" \
        -v "$(pwd)/test_outbox:/workspace/test_outbox" \
        -v "$(pwd)/test_logs:/workspace/test_logs" \
        -v "$(pwd)/test_georeferenced:/workspace/test_georeferenced" \
        -v "$(pwd)/test_postprocessed:/workspace/test_postprocessed" \
        -v "$(pwd)/test_denoising:/workspace/test_denoising" \
        -v "$(pwd)/test_results:/workspace/test_results" \
        -p 6006:6006 \
        sar-ship-detection:latest
else
    echo "No NVIDIA GPU detected or nvidia-smi not available, running in CPU-only mode..."
    # Run the container without GPU support
    docker run -it --rm \
        --ipc=host \
        --name sar-ship-detection \
        -v "$(pwd)/data:/workspace/data:ro" \
        -v "$(pwd)/weights:/workspace/weights:ro" \
        -v "$(pwd)/configs:/workspace/configs:ro" \
        -v "$(pwd)/test_ingest:/workspace/test_ingest" \
        -v "$(pwd)/test_work:/workspace/test_work" \
        -v "$(pwd)/test_detections:/workspace/test_detections" \
        -v "$(pwd)/test_thumbs:/workspace/test_thumbs" \
        -v "$(pwd)/test_outbox:/workspace/test_outbox" \
        -v "$(pwd)/test_logs:/workspace/test_logs" \
        -v "$(pwd)/test_georeferenced:/workspace/test_georeferenced" \
        -v "$(pwd)/test_postprocessed:/workspace/test_postprocessed" \
        -v "$(pwd)/test_denoising:/workspace/test_denoising" \
        -v "$(pwd)/test_results:/workspace/test_results" \
        -p 6006:6006 \
        sar-ship-detection:latest
fi

echo "Container stopped."