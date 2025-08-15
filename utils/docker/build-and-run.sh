#!/bin/bash

# Build and Run Script for SAR Ship Detection Pipeline
# This script builds the Docker image and runs the container

set -e

echo "🚀 Building SAR Ship Detection Pipeline Docker Image..."

# Build the image
docker build -f utils/docker/Dockerfile-laptop -t sar-ship-detection:latest ..

echo "✅ Image built successfully!"

echo "🐳 Running container..."

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

echo "🏁 Container stopped."