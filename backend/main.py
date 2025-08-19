# main.py
import io
import os
import cv2
import torch
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import List

# --- HELPER FUNCTIONS (To simulate preprocessing) ---

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes and adds padding (letterbox) to an image to fit a new shape
    without distorting the aspect ratio. This is a visual simulation of YOLO's preprocessing.
    """
    shape = img.shape[:2]  # current height, width
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate scale ratio and new unpadded size
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding

    # Split padding into two sides
    dw /= 2
    dh /= 2

    # Resize image
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add border (padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img

def image_to_base64(img_np: np.ndarray) -> str:
    """Encodes a NumPy image (BGR) into a base64 string."""
    _, buffer = cv2.imencode('.jpg', img_np)
    return base64.b64encode(buffer).decode('utf-8')

# --- 1. Model Loading and Configuration ---
YOLO5_REPO_PATH = 'yolov5'
MODEL_PATH = 'best.pt'

print("Loading YOLOv5 model...")
model = torch.hub.load(YOLO5_REPO_PATH, 'custom', path=MODEL_PATH, source='local')
model.conf = 0.25
model.iou = 0.45

if torch.cuda.is_available():
    model.to('cuda')
    print(f"✅ Model '{MODEL_PATH}' loaded successfully on GPU.")
else:
    print(f"✅ Model '{MODEL_PATH}' loaded successfully on CPU.")


# --- 2. FastAPI App Creation ---
app = FastAPI(title="YOLOv5 Ship Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


# --- 3. Prediction Endpoint ---
@app.post("/predict/")
async def predict_image(options: List[str] = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    
    # Decode the image once to use it in all steps
    image_np_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(image_np_rgb)
    df = results.pandas().xyxy[0]
    
    # Prepare the response structure
    response_data = {
        "images": {},
        "count_result": None,
        "probabilities_result": None,
    }

    # --- Process and generate the requested responses ---

    if 'Number of Ships' in options:
        response_data['count_result'] = len(df)

    if 'Detection Probabilities' in options:
        response_data['probabilities_result'] = df['confidence'].tolist()

    if 'Input Image' in options:
        # Simply encode the original bytes we already have
        response_data['images']['input'] = base64.b64encode(contents).decode('utf-8')
    
    if 'Preprocessed Image' in options:
        # Apply letterboxing to visualize preprocessing
        preprocessed_img = letterbox(image_np_bgr)
        response_data['images']['preprocessed'] = image_to_base64(preprocessed_img)

    if 'Final Image with Detections' in options:
        # Use YOLOv5's efficient rendering
        rendered_img_rgb = results.render()[0]
        rendered_img_bgr = cv2.cvtColor(rendered_img_rgb, cv2.COLOR_RGB2BGR)
        response_data['images']['final'] = image_to_base64(rendered_img_bgr)

    return response_data

@app.get("/")
def read_root():
    return {"status": "OK", "message": "YOLOv5 ship detection API is online."}