#!/bin/bash

# Jetson Performance Test Script
# Tests different inference scenarios and benchmarks performance
# Author: @amanarora9848

set -e

echo "======================================================"
echo "Jetson AGX Orin Performance Test Suite"
echo "======================================================"

# Check if we're on Jetson
if [ "$(uname -m)" = "aarch64" ]; then
    echo "✅ Running on ARM64 architecture (Jetson compatible)"
    
    # Check for Jetson-specific files
    if [ -f "/etc/nv_tegra_release" ] || [ -f "/sys/module/tegra_fuse/parameters/tegra_chip_id" ]; then
        echo "✅ Jetson hardware detected"
        IS_JETSON=true
    else
        echo "⚠️  ARM64 architecture but no Jetson hardware detected"
        IS_JETSON=false
    fi
else
    echo "⚠️  Running on x86_64 architecture (testing Jetson compatibility)"
    IS_JETSON=false
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    GPU_AVAILABLE=true
else
    echo "⚠️  No NVIDIA GPU detected - will test CPU only"
    GPU_AVAILABLE=false
fi

echo ""
echo "======================================================"
echo "Test 1: Jetson Compatibility Check"
echo "======================================================"

python test_jetson_compatibility.py

echo ""
echo "======================================================"
echo "Test 2: CPU Inference Performance"
echo "======================================================"

# Ensure we have test images with proper permissions
if [ ! -d "test_ingest" ] || [ -z "$(ls -A test_ingest 2>/dev/null)" ]; then
    echo "Preparing test images..."
    mkdir -p test_ingest
    cp source/* test_ingest/ 2>/dev/null || echo "No source images found"
fi

# Ensure proper ownership of test directories
echo "Setting up test directories with proper permissions..."
mkdir -p test_work test_detections test_thumbs test_outbox test_logs test_metadata test_georeferenced test_postprocessed

# Test CPU inference
echo "Running CPU inference test..."
python detect_jetson.py \
    --weights runs/train/experiment4/weights/best.pt \
    --source test_ingest/ \
    --device cpu \
    --nosave \
    --project runs/test_jetson \
    --name cpu_test \
    --exist-ok

if [ "$GPU_AVAILABLE" = true ]; then
    echo ""
    echo "======================================================"
    echo "Test 3: GPU Inference Performance (FP32)"
    echo "======================================================"
    
    # Test GPU FP32 inference
    echo "Running GPU FP32 inference test..."
    python detect_jetson.py \
        --weights runs/train/experiment4/weights/best.pt \
        --source test_ingest/ \
        --device 0 \
        --nosave \
        --project runs/test_jetson \
        --name gpu_fp32_test \
        --exist-ok

    echo ""
    echo "======================================================"
    echo "Test 4: GPU Inference Performance (FP16)"
    echo "======================================================"
    
    # Test GPU FP16 inference (Jetson optimized)
    echo "Running GPU FP16 inference test (Jetson optimized)..."
    python detect_jetson.py \
        --weights runs/train/experiment4/weights/best.pt \
        --source test_ingest/ \
        --device 0 \
        --half \
        --nosave \
        --project runs/test_jetson \
        --name gpu_fp16_test \
        --exist-ok

    if [ "$IS_JETSON" = true ]; then
        echo ""
        echo "======================================================"
        echo "Test 5: ONNX Model Performance (Jetson)"
        echo "======================================================"
        
        # Test ONNX model if available
        if [ -f "jetson_export/model.onnx" ]; then
            echo "Running ONNX model test..."
            python detect_jetson.py \
                --weights jetson_export/model.onnx \
                --source test_ingest/ \
                --device 0 \
                --half \
                --nosave \
                --project runs/test_jetson \
                --name onnx_test \
                --exist-ok
        else
            echo "⚠️  ONNX model not found. Run: python export_jetson.py --include onnx"
        fi

        echo ""
        echo "======================================================"
        echo "Test 6: TensorRT Engine Performance (Jetson)"
        echo "======================================================"
        
        # Test TensorRT engine if available
        if [ -f "jetson_export/model.engine" ]; then
            echo "Running TensorRT engine test..."
            python detect_jetson.py \
                --weights jetson_export/model.engine \
                --source test_ingest/ \
                --device 0 \
                --half \
                --nosave \
                --project runs/test_jetson \
                --name tensorrt_test \
                --exist-ok
        else
            echo "⚠️  TensorRT engine not found. Run: python export_jetson.py --include engine"
        fi
    fi
fi

echo ""
echo "======================================================"
echo "Test 7: Full Pipeline Demo"
echo "======================================================"

# Refresh test images for demo with proper permissions
echo "Refreshing test images for pipeline demo..."
cp source/* test_ingest/ 2>/dev/null || echo "No source images to copy"

# Ensure all test directories have proper ownership
echo "Ensuring proper directory permissions for pipeline demo..."
chown -R $(whoami):$(whoami) test_ingest test_work test_detections test_thumbs test_outbox test_logs test_metadata test_georeferenced test_postprocessed 2>/dev/null || true

# Run the full demo
echo "Running full pipeline demo..."
python demo.py

echo ""
echo "======================================================"
echo "Test Results Summary"
echo "======================================================"

# Show test results
if [ -d "runs/test_jetson" ]; then
    echo "Test results saved in: runs/test_jetson/"
    ls -la runs/test_jetson/
    
    echo ""
    echo "Performance comparison:"
    for test_dir in runs/test_jetson/*/; do
        if [ -d "$test_dir" ]; then
            test_name=$(basename "$test_dir")
            echo "  $test_name: $(ls -1 "$test_dir" 2>/dev/null | wc -l) output files"
        fi
    done
else
    echo "No test results found"
fi

echo ""
echo "======================================================"
echo "Jetson Deployment Recommendations"
echo "======================================================"

if [ "$IS_JETSON" = true ]; then
    echo "✅ Running on Jetson - Deployment Ready!"
    echo ""
    echo "Recommended commands for production:"
    echo "  1. Full pipeline: python demo.py"
    echo "  2. Direct inference: python detect_jetson.py --weights runs/train/experiment4/weights/best.pt --source /path/to/images --device 0 --half"
    
    if [ -f "jetson_export/model.engine" ]; then
        echo "  3. TensorRT (fastest): python detect_jetson.py --weights jetson_export/model.engine --source /path/to/images --device 0 --half"
    fi
elif [ "$GPU_AVAILABLE" = true ]; then
    echo "✅ GPU available - Compatible for testing"
    echo ""
    echo "For Jetson deployment:"
    echo "  1. Transfer this code to Jetson"
    echo "  2. Build Docker image: ./utils/docker/build-jetson.sh"
    echo "  3. Run with GPU: sudo docker run --runtime nvidia --gpus all sar-ship-detection:jetson"
else
    echo "⚠️  CPU only - Limited performance"
    echo ""
    echo "For better performance:"
    echo "  1. Deploy on Jetson with GPU"
    echo "  2. Enable CUDA acceleration"
    echo "  3. Use FP16 precision"
fi

echo ""
echo "Test complete! 🚀"