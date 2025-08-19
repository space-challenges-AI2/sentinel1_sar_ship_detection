#!/usr/bin/env python3
"""
Simple test script to verify YOLOv5 detection works in Docker
"""

import os
import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_detection():
    """Test basic YOLOv5 detection functionality"""
    try:
        print("Testing YOLOv5 detection...")
        
        # Test if we can import YOLOv5
        from models.common import DetectMultiBackend
        from utils.general import check_img_size, non_max_suppression, scale_boxes
        from utils.torch_utils import select_device
        from utils.augmentations import letterbox
        import torch
        import cv2
        import numpy as np
        
        print("✓ YOLOv5 imports successful")
        
        # Check if weights exist
        weights_path = "weights/yolov5n.pt"
        if not os.path.exists(weights_path):
            print(f"✗ Weights not found at {weights_path}")
            return False
        
        print("✓ Weights file found")
        
        # Test device selection
        device = select_device('auto')
        print(f"✓ Device selected: {device}")
        
        # Test model loading
        model = DetectMultiBackend(weights_path, device=device)
        print("✓ Model loaded successfully")
        
        # Test inference on a simple image
        img_size = (640, 640)
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Preprocess image
        img = letterbox(img, img_size, stride=model.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        
        # Inference
        pred = model(img)
        print("✓ Inference successful")
        
        # Post-process
        pred = non_max_suppression(pred, 0.25, 0.45, max_det=1000)
        print("✓ Post-processing successful")
        
        print("🎉 All tests passed! YOLOv5 detection is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detection()
    sys.exit(0 if success else 1) 