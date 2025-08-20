# Jetson AGX Orin Deployment Guide

## Overview

This guide covers deploying the SAR Ship Detection Pipeline on **NVIDIA Jetson AGX Orin** for edge inference. The deployment is optimized for ARM64 architecture and provides TensorRT acceleration.

## Prerequisites

### Hardware Requirements
- **NVIDIA Jetson AGX Orin** (32GB RAM recommended)
- **MicroSD card** (64GB+ Class 10) or **NVMe SSD**
- **Power supply** (65W+ recommended for full performance)

### Software Requirements
- **JetPack 5.1.1** or later
- **Docker** with NVIDIA runtime
- **NVIDIA Container Toolkit**

## Quick Start

### 1. Export Model for Jetson

First, export your trained model to Jetson-optimized formats:

```bash
# Export experiment4 model to Jetson formats
python export_jetson.py --weights runs/train/experiment4/weights/best.pt

# This creates:
# - jetson_export/model.onnx (ONNX format)
# - jetson_export/model.engine (TensorRT format) 
# - jetson_export/model_jetson_optimized.pt (PyTorch optimized)
```

### 2. Build Jetson Docker Image

```bash
# Make build script executable
chmod +x utils/docker/build-jetson.sh

# Build Jetson-optimized image
./utils/docker/build-jetson.sh
```

### 3. Run on Jetson

```bash
# Run with GPU access
sudo docker run --runtime nvidia --gpus all --network host --privileged -it \
  -v $(pwd):/workspace \
  sar-ship-detection:jetson

# Inside container, run the demo
python demo.py

# Or test detection directly
python detect_jetson.py --source test_ingest/ --weights runs/train/experiment4/weights/best.pt --device 0
```

## Detailed Deployment Steps

### Step 1: Prepare Jetson Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Step 2: Transfer Model Files

```bash
# Copy exported models to Jetson
scp -r jetson_export/ ubuntu@jetson-ip:/home/ubuntu/projects/sar-ship-detection/

# Or use USB drive/SD card
cp -r jetson_export/ /media/ubuntu/usb-drive/
```

### Step 3: Build and Run Container

```bash
# Navigate to project directory
cd ~/projects/sar-ship-detection

# Build Jetson image
./utils/docker/build-jetson.sh

# Run container with full access
sudo docker run --runtime nvidia --gpus all --network host --privileged -it \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  sar-ship-detection:jetson
```

## Performance Optimization

### 1. TensorRT Acceleration

For maximum performance, use the TensorRT engine:

```bash
# Inside container, use TensorRT model
python detect_jetson.py \
  --weights jetson_export/model.engine \
  --source test_ingest/ \
  --device 0 \
  --half  # Enable FP16
```

### 2. Memory Optimization

```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Set optimal power mode
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Max clock speeds
```

### 3. Batch Processing

```bash
# Process multiple images efficiently
python detect_jetson.py \
  --weights runs/train/experiment4/weights/best.pt \
  --source test_ingest/ \
  --device 0 \
  --batch-size 4  # Adjust based on available memory
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size or image resolution
python detect_jetson.py --batch-size 1 --imgsz 512

# Check available memory
nvidia-smi
```

#### 2. Docker Permission Issues
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo docker run --runtime nvidia --gpus all ...
```

#### 3. Model Loading Errors
```bash
# Verify model file exists
ls -la runs/train/experiment4/weights/

# Check model compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Performance Issues
```bash
# Check power mode
sudo nvpmodel -q

# Set to max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor temperatures
tegrastats
```

### Performance Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# System resource monitoring
htop

# Jetson-specific stats
tegrastats

# Temperature monitoring
cat /sys/class/thermal/thermal_zone*/temp
```

## Benchmarking

### Performance Comparison

| Format | Precision | Memory | Speed | Notes |
|--------|-----------|---------|-------|-------|
| PyTorch | FP32 | ~2GB | Baseline | Standard inference |
| PyTorch | FP16 | ~1.5GB | 1.5x | Jetson optimized |
| ONNX | FP32 | ~1.8GB | 1.8x | Open standard |
| TensorRT | FP16 | ~1.2GB | 2.5x | **Recommended** |

### Run Benchmarks

```bash
# Benchmark PyTorch model
python detect_jetson.py \
  --weights runs/train/experiment4/weights/best.pt \
  --source test_ingest/ \
  --device 0

# Benchmark TensorRT model
python detect_jetson.py \
  --weights jetson_export/model.engine \
  --source test_ingest/ \
  --device 0 \
  --half
```

## Production Deployment

### 1. Service Configuration

```bash
# Create systemd service
sudo nano /etc/systemd/system/sar-detection.service

[Unit]
Description=SAR Ship Detection Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker run --runtime nvidia --gpus all --name sar-detection \
  -v /opt/sar-detection:/workspace \
  sar-ship-detection:jetson
ExecStop=/usr/bin/docker stop sar-detection
ExecStopPost=/usr/bin/docker rm sar-detection

[Install]
WantedBy=multi-user.target
```

### 2. Auto-start Service

```bash
# Enable service
sudo systemctl enable sar-detection.service

# Start service
sudo systemctl start sar-detection.service

# Check status
sudo systemctl status sar-detection.service
```

### 3. Logging and Monitoring

```bash
# View service logs
sudo journalctl -u sar-detection.service -f

# Monitor resource usage
watch -n 5 'nvidia-smi && echo "---" && df -h'
```

## Advanced Configuration

### Custom Model Paths

```bash
# Use custom model location
python demo.py --weights /opt/models/sar_ship_detection.pt

# Or modify demo.py
coordinator.weights_path = Path("/opt/models/sar_ship_detection.pt")
```

### Environment Variables

```bash
# Set in Docker run command
sudo docker run -e CUDA_VISIBLE_DEVICES=0 \
  -e OMP_NUM_THREADS=4 \
  -e TORCH_CUDNN_BENCHMARK=1 \
  sar-ship-detection:jetson
```

### Network Configuration

```bash
# Run with host networking for better performance
sudo docker run --network host \
  --runtime nvidia --gpus all \
  sar-ship-detection:jetson
```

## Support and Resources

### Useful Commands

```bash
# Check Jetson version
cat /etc/nv_tegra_release

# Check CUDA version
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"

# Check TensorRT
python -c "import tensorrt as trt; print(trt.__version__)"
```

### Documentation Links

- [Jetson AGX Orin Developer Guide](https://developer.nvidia.com/embedded/jetson-agx-orin)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Jetson](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)

### Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your Jetson environment setup
3. Check Docker and NVIDIA runtime installation
4. Monitor system resources and temperatures
5. Review container logs for error messages

## Next Steps

After successful deployment:

1. **Optimize performance** with TensorRT
2. **Set up monitoring** and logging
3. **Configure auto-restart** for production
4. **Implement batch processing** for multiple images
5. **Add health checks** and alerting

Your SAR ship detection pipeline is now ready for Jetson AGX Orin deployment! 🚀 