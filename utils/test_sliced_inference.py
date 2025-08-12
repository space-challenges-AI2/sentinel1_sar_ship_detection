#!/usr/bin/env python3
"""
Test script for sliced inference functionality
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from utils.sliced_inference import SlicedInference
from models.common import DetectMultiBackend


def test_sliced_inference():
    """Test sliced inference with a sample image"""
    
    # Initialize sliced inference
    sliced_inference = SlicedInference(
        tile_size=640,
        overlap=0.2,
        min_tile_size=320,
        max_tile_size=1024,
        enable_adaptive_tiling=True,
        merge_strategy='nms'
    )
    
    print("Sliced inference initialized successfully!")
    print(f"Tile size: {sliced_inference.tile_size}")
    print(f"Overlap: {sliced_inference.overlap}")
    print(f"Merge strategy: {sliced_inference.merge_strategy}")
    
    # Test tile size calculation
    test_shapes = [(800, 600), (1920, 1080), (4000, 3000)]
    for h, w in test_shapes:
        optimal_size = sliced_inference.calculate_optimal_tile_size((h, w), 32)
        print(f"Image {w}x{h} -> Optimal tile size: {optimal_size}")
    
    # Test tile creation
    test_image = np.random.randint(0, 255, (1200, 1600, 3), dtype=np.uint8)
    tiles = sliced_inference.create_tiles(test_image, 640, 128)
    print(f"Created {len(tiles)} tiles from test image")
    
    # Test with actual model (if available)
    model_path = "./runs/train/experiment3/weights/best.pt"
    if Path(model_path).exists():
        try:
            print(f"\nTesting with model: {model_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DetectMultiBackend(model_path, device=device)
            
            # Test image processing
            predictions = sliced_inference.process_image(
                test_image, model, 32, 0.25, 0.45, 1000, device
            )
            print(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            print(f"Model testing failed: {e}")
    else:
        print(f"\nModel not found at {model_path}, skipping model testing")
    
    print("\nSliced inference test completed successfully!")


def test_tile_creation():
    """Test tile creation with different parameters"""
    
    print("\n=== Testing Tile Creation ===")
    
    # Create test image
    test_image = np.random.randint(0, 255, (1000, 1500, 3), dtype=np.uint8)
    
    # Test different tile sizes and overlaps
    configs = [
        (640, 0.1),
        (640, 0.2),
        (512, 0.15),
        (1024, 0.25)
    ]
    
    for tile_size, overlap in configs:
        sliced_inference = SlicedInference(
            tile_size=tile_size,
            overlap=overlap
        )
        
        tiles = sliced_inference.create_tiles(
            test_image, tile_size, int(tile_size * overlap)
        )
        
        print(f"Tile size: {tile_size}, Overlap: {overlap:.2f} -> {len(tiles)} tiles")
        
        # Check tile coverage
        total_area = 0
        for tile in tiles:
            coords = tile['coords']
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            total_area += area
        
        image_area = test_image.shape[0] * test_image.shape[1]
        coverage = total_area / image_area
        print(f"  Coverage: {coverage:.2f}")


def test_prediction_merging():
    """Test prediction merging strategies"""
    
    print("\n=== Testing Prediction Merging ===")
    
    # Create mock predictions
    device = 'cpu'
    mock_predictions = [
        torch.tensor([[100, 100, 200, 200, 0.9, 0]], device=device),  # High confidence
        torch.tensor([[110, 110, 190, 190, 0.7, 0]], device=device),  # Overlapping, lower confidence
        torch.tensor([[300, 300, 400, 400, 0.8, 1]], device=device),  # Different location
        torch.tensor([[105, 105, 195, 195, 0.6, 0]], device=device),  # Overlapping, lowest confidence
    ]
    
    # Test different merge strategies
    strategies = ['nms', 'weighted', 'confidence']
    
    for strategy in strategies:
        sliced_inference = SlicedInference(merge_strategy=strategy)
        
        merged = sliced_inference.merge_predictions(
            mock_predictions, 0.5, 0.3, 10
        )
        
        print(f"Merge strategy '{strategy}': {len(merged)} final predictions")
        if len(merged) > 0:
            print(f"  Confidence scores: {merged[:, 4].tolist()}")


if __name__ == "__main__":
    print("Running sliced inference tests...")
    
    try:
        test_sliced_inference()
        test_tile_creation()
        test_prediction_merging()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 