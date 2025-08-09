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
