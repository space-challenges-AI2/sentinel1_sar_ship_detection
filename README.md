# WaveTrack.AI
### Lightweight Inferencing for accurate and fast Vessels-in-water detections and classification, developed during Space Challenges 2025 in Sofia, Bulgaria. It contains the code for Ship Detection from Sentinel-1 SAR Images.

The code has been inspired and some parts adopted from [LEAD-YOLO](https://github.com/qingqing-zijin/LEAD-YOLO/tree/main?tab=readme-ov-file), licensed under the GNU General Public License v3.0.


### Setup:

#### 1. Install Dependencies

Ensure your system meets the following requirements:
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.3 (for GPU support)

#### 2. Clone the Repository

```bash
git clone https://github.com/space-challenges-AI2/sentinel1_sar_ship_detection.git
cd sentinel1_sar_ship_detection
```

#### 3. Create your favourite environmnent (python venv or conda)
- For venv:
```bash
python3 -m venv .venv
```
- For conda env
```bash
conda create -n sc25 python=3.11
```

#### 4. Install requirements
```bash
pip install -r requirements.txt
```

#### 5. Train a model!

**Basic training:**
```bash
python train.py --cfg models/yolov5n.yaml --data data/HRSID_land.yaml --hyp data/hyp/hyp.scratch-low.yaml --weights '' --epochs 350
```

**Training with custom experiment name:**
```bash
python train.py --cfg models/yolov5n.yaml --data data/HRSID_land.yaml --hyp data/hyp/hyp.scratch-low.yaml --weights '' --epochs 350 --name my_experiment
```

**Training with custom project and name:**
```bash
python train.py --cfg models/yolov5n.yaml --data data/HRSID_land.yaml --hyp data/hyp/hyp.scratch-low.yaml --weights '' --epochs 350 --project my_project --name my_experiment
```

**Note:** If you don't specify a `--name`, experiments will be named `experiment`, `experiment2`, `experiment3`, etc. If you specify a custom name that already exists, it will automatically append a number (e.g., `my_experiment2`, `my_experiment3`).

The corresponding training files for the experiment are saved inside `runs/train/experiment{i}`, including the weights.

#### 6. Evaluate the model
```bash
python val.py --weights runs/train/experiment/weights/best.pt --data data/HRSID_land.yaml --img 640 --batch-size 32
```

**Note:** Replace `experiment` with your actual experiment name if you used a custom name during training.

The corresponding validation files for the experiment are saved inside `runs/train/experiment{i}`, including the graphs and metrics for the experiments.


#### 7. Inference

Run inference on a single image or a batch of images:

```bash
python detect.py --source ./source --weights runs/train/lead_yolo4/weights/best.pt
```
The inference outputs (e.g., annotated images or videos) are saved in `runs/detect/`.

#### 8. Docker

WaveTrack.AI provides Docker support for containerized deployment, including specialized configurations for satellite and space applications. The project includes two main Dockerfile variants:

##### **Available Docker Images**

1. **Standard Docker Image** (`utils/docker/Dockerfile`)
   - Based on PyTorch with CUDA support
   - Optimized for GPU-accelerated inference
   - Suitable for ground-based processing and development

2. **ARM64 Docker Image** (`utils/docker/Dockerfile-arm64`)
   - Based on Ubuntu ARM64
   - Compatible with ARM architectures (Jetson Nano, Raspberry Pi, Apple M1)
   - **Recommended for satellite deployment** due to power efficiency and space constraints

##### **Quick Start**

**Build the standard image:**
```bash
# Build for x86_64 with GPU support
docker build -f utils/docker/Dockerfile -t wavetrack-ai:latest .
```

**Build the ARM64 image (recommended for satellites):**
```bash
# Build for ARM64 architecture
docker build --platform linux/arm64 -f utils/docker/Dockerfile-arm64 -t wavetrack-ai:arm64 .
```

**Run the container:**
```bash
# Run with GPU support (if available)
docker run -it --gpus all -v $(pwd)/data:/usr/src/app/data wavetrack-ai:latest

# Run ARM64 version (CPU only)
docker run -it -v $(pwd)/data:/usr/src/app/data wavetrack-ai:arm64
```

## 🛰️ Pipeline Architecture & Design

### Overview
WaveTrack.AI implements a comprehensive SAR (Synthetic Aperture Radar) ship detection pipeline that processes satellite imagery through multiple stages to identify and classify vessels in water bodies.

### Pipeline Components

#### 1. **Ingest Service** (`SARIngestService`)
- **Purpose**: Monitors and manages incoming SAR image tiles
- **Functionality**: 
  - Watches for new SAR data files
  - Queues images for processing
  - Manages work item lifecycle
- **Output**: Queued work items for pipeline processing

#### 2. **Detection Service** (YOLO-based)
- **Purpose**: Performs AI-powered ship detection on SAR images
- **Features**:
  - YOLOv5-based object detection
  - Built-in denoising capabilities (FABF, None)
  - GPU/CPU acceleration support
  - Configurable confidence thresholds
- **Output**: Detection results with bounding boxes and confidence scores

#### 3. **Georeferencing Service** (`GeoreferencingService`)
- **Purpose**: Converts pixel coordinates to geographic coordinates
- **Functionality**:
  - Transforms detection coordinates to lat/long
  - Handles SAR image geolocation metadata
  - Provides geographic context for detections
- **Output**: Georeferenced detection coordinates

#### 4. **Post-Processing Service** (`PostProcessingService`)
- **Purpose**: Enhances and validates detection results
- **Features**:
  - Generates thumbnail images
  - Applies post-processing filters
  - Quality assessment and validation
- **Output**: Enhanced detection results with thumbnails

#### 5. **Packager Service** (`PackagerService`)
- **Purpose**: Creates downlink packets for satellite transmission
- **Functionality**:
  - Bundles detection results
  - Optimizes data for transmission
  - Creates standardized output formats
- **Output**: Transmission-ready data packets

#### 6. **Health Monitor** (`HealthMonitor`)
- **Purpose**: Monitors pipeline health and performance
- **Features**:
  - Real-time status monitoring
  - Performance metrics collection
  - Error logging and alerting
- **Output**: Health status and performance reports

### Pipeline Flow

```
SAR Image Input → Ingest → Detection → Georeferencing → Post-Processing → Packaging → Output
     ↓              ↓         ↓           ↓              ↓              ↓         ↓
  test_ingest/  test_work/  test_detections/  test_georeferenced/  test_postprocessed/  test_outbox/
```

### Directory Structure
The pipeline creates and manages the following test directories:
- `test_ingest/` - Input SAR images
- `test_work/` - Intermediate processing files
- `test_metadata/` - Image and processing metadata
- `test_detections/` - YOLO detection results
- `test_thumbs/` - Generated thumbnail images
- `test_outbox/` - Final output packets
- `test_logs/` - Processing logs
- `test_georeferenced/` - Georeferenced coordinates
- `test_postprocessed/` - Enhanced detection results
- `test_denoising/` - Denoising artifacts
- `test_results/` - Additional processing results

## 🐳 Docker Setup & Pipeline Demo

### Build the Docker Image
```bash
sudo docker build -f utils/docker/Dockerfile-laptop -t sar-ship-detection:latest .
```

### Run the Pipeline Demo
```bash
sudo docker run -it --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  --entrypoint python \
  sar-ship-detection:latest demo.py
```

### What Happens
1. Container starts with your project mounted to `/workspace`
2. Pipeline processes images from `source/` directory
3. All outputs are saved to your local test directories
4. When container stops, all data remains in your local filesystem

### Expected Outputs
- `test_detections/pipeline/` - YOLO detection results
- `test_thumbs/` - Generated thumbnails  
- `test_metadata/` - Processing information
- `test_outbox/` - Final output packets
- `test_logs/` - Processing logs

## 🔧 Pipeline Development & Customization

### Pipeline Coordinator
The main pipeline orchestration is handled by `PipelineCoordinator` in `utils/pipeline/coordinator.py`. This class:
- Manages the entire pipeline workflow
- Coordinates between different services
- Handles error recovery and monitoring
- Provides real-time status updates

### Customizing the Pipeline
You can modify the pipeline behavior by:
- Adjusting denoising parameters in the coordinator
- Modifying service configurations
- Adding new processing stages
- Customizing output formats

### Configuration Files
Pipeline behavior is controlled by:
- `configs/flight.env` - Environment-specific settings
- Service-specific configuration parameters
- Runtime command-line arguments

## Troubleshooting

### Common Issues

1. **Container can't find entrypoint.sh**
   - Solution: Use `--entrypoint python` or `--entrypoint bash`

2. **No output files in local directory**
   - Solution: Ensure you're using `-v $(pwd):/workspace` volume mount

3. **GPU not detected**
   - Solution: Install NVIDIA Docker runtime and use `--gpus all`

4. **Permission denied errors**
   - Solution: Use `sudo` for Docker commands

### Debug Commands

```bash
# Check Docker image exists
sudo docker images | grep sar-ship-detection

# Check container logs
sudo docker logs <container_id>

# Interactive debugging
sudo docker run -it --rm --gpus all --ipc=host -v $(pwd):/workspace --entrypoint bash sar-ship-detection:latest
```

## Performance & Monitoring

### Real-time Monitoring
The pipeline provides real-time status updates including:
- Images processed count
- Detection accuracy metrics
- Processing time statistics
- System health status

### Health Checks
- Pipeline component status
- Resource utilization
- Error rate monitoring
- Performance metrics


## Jetson AGX Orin Deployment

### Overview
WaveTrack.AI has been optimized for deployment on NVIDIA Jetson AGX Orin devices, providing real-time SAR ship detection capabilities in edge computing environments. The Jetson deployment includes specialized optimizations for ARM64 architecture, TensorRT acceleration, and power-efficient inference.

### Prerequisites

#### Hardware Requirements
- NVIDIA Jetson AGX Orin (8GB, 16GB, or 32GB variants)
- MicroSD card (64GB minimum, Class 10 recommended)
- Power supply (65W or higher)
- USB-C cable for development

#### Software Requirements
- JetPack 5.1 or later
- Python 3.8+
- CUDA 11.4+
- cuDNN 8.6+
- TensorRT 8.5+

### Jetson Setup

#### 1. Flash JetPack
```bash
# Download JetPack 5.1+ from NVIDIA Developer Portal
# Use NVIDIA SDK Manager to flash your Jetson device
# Ensure CUDA, cuDNN, and TensorRT are included in the installation
```

#### 2. Install System Dependencies
```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio
```

#### 3. Install Python Dependencies
```bash
# Create Python virtual environment
python3 -m venv jetson_env
source jetson_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Jetson (ARM64)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### Jetson-Specific Optimizations

#### 1. Model Export for Jetson
```bash
# Export PyTorch model to ONNX format
python export_jetson.py \
    --weights runs/train/experiment4/weights/best.pt \
    --include onnx \
    --img 640 \
    --batch 1

# Export to TensorRT engine (recommended for production)
python export_jetson.py \
    --weights runs/train/experiment4/weights/best.pt \
    --include engine \
    --img 640 \
    --batch 1 \
    --half
```

#### 2. Jetson Performance Tuning
```bash
# Set Jetson performance mode
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Enable all clocks

# Monitor GPU status
nvidia-smi
tegrastats
```

### Deployment Methods

#### Method 1: Direct Deployment (Recommended)

#### 1. Clone Repository on Jetson
```bash
git clone https://github.com/space-challenges-AI2/sentinel1_sar_ship_detection.git
cd sentinel1_sar_ship_detection
```

#### 2. Setup Environment
```bash
# Create and activate conda environment
conda create -n sc25 python=3.11
conda activate sc25

# Install dependencies
pip install -r requirements.txt

# Install Jetson-specific packages
pip install opencv-python-headless>=4.5.0
pip install watchdog>=3.0.0
```

#### 3. Test Jetson Compatibility
```bash
# Run compatibility check
python test_jetson_compatibility.py

# Run performance test suite
./test_jetson_performance.sh
```

#### 4. Run Inference
```bash
# CPU inference
python detect_jetson.py \
    --weights runs/train/experiment4/weights/best.pt \
    --source /path/to/images \
    --device cpu \
    --img 640

# GPU inference (FP32)
python detect_jetson.py \
    --weights runs/train/experiment4/weights/best.pt \
    --source /path/to/images \
    --device 0 \
    --img 640

# GPU inference (FP16 - recommended for Jetson)
python detect_jetson.py \
    --weights runs/train/experiment4/weights/best.pt \
    --source /path/to/images \
    --device 0 \
    --half \
    --img 640

# TensorRT inference (fastest)
python detect_jetson.py \
    --weights jetson_export/model.engine \
    --source /path/to/images \
    --device 0 \
    --half \
    --img 640
```

#### 5. Run Full Pipeline
```bash
# Run the complete pipeline demo
python demo.py
```

#### Method 2: Docker Deployment

#### 1. Build Jetson Docker Image
```bash
# Build optimized Docker image for Jetson
./utils/docker/build-jetson.sh

# Or manually build
sudo docker build -f utils/docker/Dockerfile-jetson -t sar-ship-detection:jetson .
```


#### 2. Test Jetson Compatibility in Docker
```bash
# Enter the Docker container with interactive shell
sudo docker run --runtime nvidia --gpus all \
    -it --rm \
    -v $(pwd):/workspace \
    --entrypoint bash \
    sar-ship-detection:jetson

# Once inside the container, run compatibility tests:
python test_jetson_compatibility.py

# Run the full performance test suite
./test_jetson_performance.sh

# Or test individual components:
# Test CPU inference
python detect_jetson.py --weights runs/train/experiment4/weights/best.pt --source test_ingest/ --device cpu --nosave

# Test GPU inference
python detect_jetson.py --weights runs/train/experiment4/weights/best.pt --source test_ingest/ --device 0 --half --nosave

# Test TensorRT if available
python detect_jetson.py --weights jetson_export/model.engine --source test_ingest/ --device 0 --half --nosave
```


#### 3. Run Docker Container Demo
```bash
# Run with GPU access
sudo docker run --runtime nvidia --gpus all \
    -it --rm \
    -v $(pwd):/workspace \
    --entrypoint python \
    sar-ship-detection:jetson demo.py

# Run interactive shell
sudo docker run --runtime nvidia --gpus all \
    -it --rm \
    -v $(pwd):/workspace \
    --entrypoint bash \
    sar-ship-detection:jetson
```


### Performance Optimization

#### 1. Memory Management
```bash
# Monitor memory usage
free -h
nvidia-smi

# Clear GPU memory cache
sudo fuser -v /dev/nvidia*
```

#### 2. Power Management
```bash
# Check power mode
sudo nvpmodel -q

# Set power mode (0=Max, 1=5W, 2=10W, 3=15W, 4=25W, 5=30W, 6=40W, 7=50W)
sudo nvpmodel -m 0

# Enable all clocks
sudo jetson_clocks
```

#### 3. TensorRT Optimization
```bash
# Profile TensorRT engine
trtexec --loadEngine=jetson_export/model.engine \
    --iterations=100 \
    --avgRuns=10 \
    --duration=10 \
    --profilingVerbosity=detailed
```

### Production Deployment

#### 1. Systemd Service Setup
```bash
# Create service file
sudo nano /etc/systemd/system/sar-ship-detection.service
```

Add the following content:
```ini
[Unit]
Description=SAR Ship Detection Pipeline
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/sentinel1_sar_ship_detection
Environment=PATH=/home/jetson/sentinel1_sar_ship_detection/jetson_env/bin
ExecStart=/home/jetson/sentinel1_sar_ship_detection/jetson_env/bin/python demo.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2. Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable sar-ship-detection

# Start service
sudo systemctl start sar-ship-detection

# Check status
sudo systemctl status sar-ship-detection

# View logs
sudo journalctl -u sar-ship-detection -f
```

#### 3. Monitoring and Logging
```bash
# Check pipeline status
tail -f test_logs/pipeline.log

# Monitor system resources
htop
nvidia-smi -l 1

# Check disk usage
df -h
du -sh test_*
```

### Troubleshooting

#### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python detect_jetson.py --batch-size 1 --img 640
   
   # Use FP16 precision
   python detect_jetson.py --half
   
   # Clear GPU memory
   sudo fuser -v /dev/nvidia*
   ```

2. **Slow Inference Performance**
   ```bash
   # Check power mode
   sudo nvpmodel -q
   
   # Enable max performance
   sudo nvpmodel -m 0
   sudo jetson_clocks
   
   # Use TensorRT engine
   python detect_jetson.py --weights jetson_export/model.engine
   ```

3. **Permission Denied Errors**
   ```bash
   # Fix directory permissions
   sudo chown -R jetson:jetson test_*
   
   # Ensure proper ownership
   ls -la test_*
   ```

4. **Model Export Failures**
   ```bash
   # Check PyTorch version compatibility
   python -c "import torch; print(torch.__version__)"
   
   # Verify model weights
   ls -la runs/train/experiment4/weights/
   
   # Try different export formats
   python export_jetson.py --include onnx
   ```

#### Performance Benchmarks

Expected performance on Jetson AGX Orin (32GB):
- **CPU Inference**: 2-5 FPS
- **GPU FP32**: 15-25 FPS  
- **GPU FP16**: 25-40 FPS
- **TensorRT FP16**: 40-60 FPS

### Maintenance and Updates

#### 1. Regular Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
pip install --upgrade -r requirements.txt

# Update Jetson software
sudo apt install nvidia-jetpack
```

#### 2. Model Updates
```bash
# Export new model versions
python export_jetson.py --weights runs/train/new_experiment/weights/best.pt

# Restart service
sudo systemctl restart sar-ship-detection
```

#### 3. Backup and Recovery
```bash
# Backup configuration
tar -czf jetson_backup_$(date +%Y%m%d).tar.gz \
    configs/ \
    jetson_export/ \
    test_* \
    runs/

# Restore from backup
tar -xzf jetson_backup_YYYYMMDD.tar.gz
```

### Updated Project Structure

```
sentinel1_sar_ship_detection/
├── utils/pipeline/           # Pipeline orchestration
│   ├── coordinator.py       # Main pipeline coordinator
│   ├── ingest.py           # Image ingestion service
│   ├── geo.py              # Georeferencing service
│   ├── postproc.py         # Post-processing service
│   ├── packager.py         # Output packaging service
│   └── health.py           # Health monitoring
├── utils/docker/            # Docker configurations
│   ├── Dockerfile-laptop   # Optimized for x86_64
│   ├── Dockerfile-arm64    # ARM64 compatible
│   ├── Dockerfile-jetson   # Jetson optimized
│   └── build-and-run.sh    # Build automation
├── models/                  # YOLO model configurations
├── data/                    # Dataset configurations
├── weights/                 # Pre-trained model weights
├── source/                  # Test images for demo
├── demo.py                  # Main demo script
├── detect_jetson.py         # Jetson-optimized detection
├── export_jetson.py         # Jetson model export
├── test_jetson_compatibility.py  # Jetson compatibility tests
├── test_jetson_performance.sh    # Jetson performance test suite
└── requirements.txt         # Python dependencies
```
