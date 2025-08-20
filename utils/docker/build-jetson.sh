#!/bin/bash

# SAR Ship Detection Pipeline - Jetson AGX Orin Build Script
# Author: @amanarora9848

set -e

echo "Building SAR Ship Detection Pipeline for Jetson AGX Orin..."

# Check if running on Jetson
if [ "$(uname -m)" != "aarch64" ]; then
    echo "Warning: This script is designed for ARM64 architecture (Jetson)"
    echo "You are running on: $(uname -m)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the Jetson-optimized image
echo "Building Docker image..."
sudo docker build -f utils/docker/Dockerfile-jetson -t sar-ship-detection:jetson .

echo "Build complete!"
echo ""
echo "To run the container:"
echo "  sudo docker run --runtime nvidia --network host --privileged -it \\"
echo "    -v /tmp/.X11-unix:/tmp/.X11-unix \\"
echo "    -e DISPLAY=\$DISPLAY \\"
echo "    sar-ship-detection:jetson"
echo ""
echo "To run with GPU access:"
echo "  sudo docker run --runtime nvidia --gpus all --network host --privileged -it \\"
echo "    -v /tmp/.X11-unix:/tmp/.X11-unix \\"
echo "    -e DISPLAY=\$DISPLAY \\"
echo "    sar-ship-detection:jetson"
echo ""
echo "To test GPU detection:"
echo "  sudo docker run --runtime nvidia --gpus all --network host --privileged -it \\"
echo "    sar-ship-detection:jetson \\"
echo "    python detect.py --source test_ingest/ --weights runs/train/experiment4/weights/best.pt --device 0"
echo ""
echo "To run the full demo:"
echo "  sudo docker run --runtime nvidia --gpus all --network host --privileged -it \\"
echo "    -v \$(pwd):/workspace \\"
echo "    sar-ship-detection:jetson \\"
echo "    python demo.py" 