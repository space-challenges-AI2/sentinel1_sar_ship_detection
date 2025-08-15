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

## Project Structure

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
│   └── build-and-run.sh    # Build automation
├── models/                  # YOLO model configurations
├── data/                    # Dataset configurations
├── weights/                 # Pre-trained model weights
├── source/                  # Test images for demo
├── demo.py                  # Main demo script
└── requirements.txt         # Python dependencies
```
