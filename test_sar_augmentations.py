#!/usr/bin/env python3
"""
Test script for SAR augmentations
Visualizes augmented images with their labels to ensure proper transformations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import os

# Add parent directory to path to import SAR augmentations
sys.path.append(str(Path(__file__).parent))
from utils.sar_augmentations import SARAugmentations, create_synthetic_samples
import yaml

def load_sample_data():
    """Load sample images and labels from HRSID dataset"""
    # Try to find sample data
    possible_paths = [
        "data/HRSID_JPG/images/train",
        "data/HRSID_land_main/images/train",
        "data/HRSID_denoised/images/train"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            img_files = list(Path(path).glob("*.jpg"))[:3]  # Get first 3 images
            if img_files:
                return path, img_files
    
    raise FileNotFoundError("No sample images found. Please ensure dataset is available.")

def load_labels(img_path):
    """Load corresponding labels for an image"""
    # Try multiple possible label paths
    possible_label_paths = [
        img_path.parent.parent / "labels" / "train" / f"{img_path.stem}.txt",
        img_path.parent.parent / "labels" / "val" / f"{img_path.stem}.txt",
        img_path.parent.parent.parent / "labels" / "train" / f"{img_path.stem}.txt",
        img_path.parent.parent.parent / "labels" / "val" / f"{img_path.stem}.txt"
    ]
    
    for label_path in possible_label_paths:
        if label_path.exists():
            print(f"Found labels at: {label_path}")
            try:
                labels = np.loadtxt(str(label_path))
                if len(labels.shape) == 1:
                    labels = labels.reshape(1, -1)
                print(f"Loaded {len(labels)} labels from {label_path}")
                return labels
            except Exception as e:
                print(f"Error loading labels from {label_path}: {e}")
                continue
    
    print(f"No labels found for {img_path.name}")
    print("Searched paths:")
    for path in possible_label_paths:
        print(f"  {path} - {'EXISTS' if path.exists() else 'NOT FOUND'}")
    
    return np.empty((0, 5))

def draw_boxes(img, labels, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image"""
    h, w = img.shape[:2]
    img_with_boxes = img.copy()
    
    print(f"Drawing boxes on image {w}x{h}")
    
    for i, label in enumerate(labels):
        if len(label) >= 5:
            cls, x_center, y_center, width, height = label
            
            print(f"  Label {i}: class={cls}, center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
            
            # Convert normalized coordinates to pixel coordinates
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            width_px = int(width * w)
            height_px = int(height * h)
            
            print(f"  Pixel coords: center=({x_center_px}, {y_center_px}), size=({width_px}, {height_px})")
            
            # Calculate box corners
            x1 = x_center_px - width_px // 2
            y1 = y_center_px - height_px // 2
            x2 = x_center_px + width_px // 2
            y2 = y_center_px + height_px // 2
            
            print(f"  Box corners: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with coordinates for debugging
            label_text = f"Ship({x_center:.2f},{y_center:.2f})"
            cv2.putText(img_with_boxes, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)
    
    return img_with_boxes

def test_sar_augmentations():
    """Test SAR augmentations on sample images"""
    
    # Load hyperparameters
    hyp_path = Path("data/hyp/hyp.land_denoised_augmented.yaml")
    if not hyp_path.exists():
        print(f"Hyperparameters file not found: {hyp_path}")
        return
    
    with open(hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Initialize SAR augmentations
    sar_aug = SARAugmentations(hyp)
    
    # Load sample data
    try:
        data_path, img_files = load_sample_data()
        print(f"Testing with images from: {data_path}")
        
        # Debug: Check directory structure
        print(f"\nDirectory structure for {data_path}:")
        for item in Path(data_path).iterdir():
            print(f"  {item.name}")
        
        # Check if labels directory exists
        labels_dir = Path(data_path).parent.parent / "labels"
        print(f"\nLabels directory: {labels_dir} - {'EXISTS' if labels_dir.exists() else 'NOT FOUND'}")
        if labels_dir.exists():
            for item in labels_dir.iterdir():
                print(f"  {item.name}")
                if item.is_dir():
                    for subitem in item.iterdir():
                        print(f"    {subitem.name}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Test each image
    for i, img_path in enumerate(img_files):
        print(f"\n{'='*60}")
        print(f"Testing image {i+1}: {img_path.name}")
        print(f"{'='*60}")
        
        # Load image and labels
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
            
        print(f"Image loaded successfully: {img.shape}")
        
        labels = load_labels(img_path)
        print(f"Original labels: {len(labels)} ships")
        if len(labels) > 0:
            print(f"Label format: {labels}")
        
        # Apply SAR augmentations
        img_aug, labels_aug = sar_aug(img, labels)
        print(f"Augmented labels: {len(labels_aug)} ships")
        
        # Create synthetic samples
        synthetic_samples = create_synthetic_samples(img_aug, labels_aug, n_samples=2)
        print(f"Created {len(synthetic_samples)} synthetic samples")
        
        for j, (syn_img, syn_labels) in enumerate(synthetic_samples):
            print(f"  Sample {j+1}: image shape {syn_img.shape}, labels shape {syn_labels.shape}")
            if len(syn_labels) > 0:
                print(f"    Labels: {syn_labels}")
            else:
                print(f"    No valid labels!")
        
        # Calculate proper subplot layout
        n_synthetic = len(synthetic_samples)
        n_cols = max(3, n_synthetic + 2)  # At least 3 columns: original, augmented, synthetic
        
        # Create subplot layout
        fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 12))
        fig.suptitle(f'SAR Augmentation Test - {img_path.name}', fontsize=16)
        
        # Original image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_with_boxes = draw_boxes(img_rgb, labels, color=(0, 255, 0))
        axes[0, 0].imshow(img_with_boxes)
        axes[0, 0].set_title(f'Original ({len(labels)} ships)')
        axes[0, 0].axis('off')
        
        # Augmented image
        img_aug_rgb = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
        img_aug_with_boxes = draw_boxes(img_aug_rgb, labels_aug, color=(255, 0, 0))
        axes[0, 1].imshow(img_aug_with_boxes)
        axes[0, 1].set_title(f'Augmented ({len(labels_aug)} ships)')
        axes[0, 1].axis('off')
        
        # Synthetic samples
        for j, (syn_img, syn_labels) in enumerate(synthetic_samples):
            if j < n_cols - 2:
                syn_img_rgb = cv2.cvtColor(syn_img, cv2.COLOR_BGR2RGB)
                syn_img_with_boxes = draw_boxes(syn_img_rgb, syn_labels, color=(0, 0, 255))
                axes[0, 2+j].imshow(syn_img_with_boxes)
                axes[0, 2+j].set_title(f'Synthetic {j+1} ({len(syn_labels)} ships)')
                axes[0, 2+j].axis('off')
        
        # Hide unused subplots
        for j in range(2 + n_synthetic, n_cols):
            axes[0, j].axis('off')
        
        # Label comparison
        axes[1, 0].axis('off')
        if len(labels) > 0:
            axes[1, 0].text(0.1, 0.8, f'Original Labels:\n{labels}', 
                           transform=axes[1, 0].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        axes[1, 1].axis('off')
        if len(labels_aug) > 0:
            axes[1, 1].text(0.1, 0.8, f'Augmented Labels:\n{labels_aug}', 
                           transform=axes[1, 1].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        # Statistics
        axes[1, 2].axis('off')
        stats_text = f"""Augmentation Statistics:
        • Original ships: {len(labels)}
        • Augmented ships: {len(labels_aug)}
        • Synthetic samples: {len(synthetic_samples)}
        • Cutout prob: {hyp.get('cutout_prob', 0.3)}
        • Noise prob: {hyp.get('noise_prob', 0.2)}
        • Blur prob: {hyp.get('blur_prob', 0.15)}"""
        
        axes[1, 2].text(0.1, 0.8, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        # Hide unused bottom row subplots
        for j in range(3, n_cols):
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save test results
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save augmented image
        cv2.imwrite(str(output_dir / f"aug_{img_path.name}"), img_aug)
        
        # Save augmented labels
        if len(labels_aug) > 0:
            np.savetxt(str(output_dir / f"aug_{img_path.stem}.txt"), labels_aug, fmt='%.6f')
        
        # Save synthetic samples
        for j, (syn_img, syn_labels) in enumerate(synthetic_samples):
            if len(syn_labels) > 0:
                # Save synthetic image
                cv2.imwrite(str(output_dir / f"syn_{j+1}_{img_path.name}"), syn_img)
                # Save synthetic labels
                np.savetxt(str(output_dir / f"syn_{j+1}_{img_path.stem}.txt"), syn_labels, fmt='%.6f')
                print(f"Saved synthetic sample {j+1}: syn_{j+1}_{img_path.name}")
            else:
                print(f"Synthetic sample {j+1} has no valid labels - not saving")
        
        print(f"Test results saved to: {output_dir}")

def test_label_transformations():
    """Test specific label transformations"""
    print("\n" + "="*50)
    print("TESTING LABEL TRANSFORMATIONS")
    print("="*50)
    
    # Create test image and labels
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_labels = np.array([
        [0, 0.5, 0.5, 0.2, 0.1],  # Center ship
        [0, 0.2, 0.3, 0.15, 0.1],  # Top-left ship
        [0, 0.8, 0.7, 0.25, 0.15]  # Bottom-right ship
    ])
    
    print(f"Test image shape: {test_img.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Test labels:\n{test_labels}")
    
    # Test SAR augmentations
    hyp = {
        'cutout_prob': 0.5,  # High probability for testing
        'noise_prob': 0.5,
        'blur_prob': 0.5,
        'sharpen_prob': 0.5,
        'elastic_prob': 0.5
    }
    
    sar_aug = SARAugmentations(hyp)
    img_aug, labels_aug = sar_aug(test_img, test_labels)
    
    print(f"\nAfter augmentation:")
    print(f"Augmented labels shape: {labels_aug.shape}")
    print(f"Augmented labels:\n{labels_aug}")
    
    # Test synthetic samples
    synthetic_samples = create_synthetic_samples(img_aug, labels_aug, n_samples=2)
    print(f"\nSynthetic samples created: {len(synthetic_samples)}")
    
    for i, (syn_img, syn_labels) in enumerate(synthetic_samples):
        print(f"Sample {i+1} labels: {syn_labels.shape}")
        if len(syn_labels) > 0:
            print(f"Sample {i+1} labels:\n{syn_labels}")

def test_coordinate_system():
    """Test coordinate system understanding"""
    print("\n" + "="*60)
    print("TESTING COORDINATE SYSTEM")
    print("="*60)
    
    # Create a test image with known dimensions
    test_img = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # Test label: ship at center with reasonable size
    test_label = np.array([[0, 0.5, 0.5, 0.1, 0.1]])  # Center, 10% size
    
    print(f"Test label: {test_label}")
    
    # Draw box
    img_with_box = draw_boxes(test_img, test_label, color=(0, 255, 0))
    
    # Save test image
    cv2.imwrite("test_results/coordinate_test.jpg", img_with_box)
    print("Coordinate test image saved to test_results/coordinate_test.jpg")
    
    # Now test with actual label format
    actual_label = np.array([[0, 0.200625, 0.930625, 0.02625, 0.06125]])
    print(f"Actual label from data: {actual_label}")
    
    img_with_actual = draw_boxes(test_img, actual_label, color=(255, 0, 0))
    cv2.imwrite("test_results/actual_label_test.jpg", img_with_actual)
    print("Actual label test image saved to test_results/actual_label_test.jpg")

if __name__ == "__main__":
    print("SAR Augmentation Test Suite")
    print("="*50)
    
    # Test coordinate system first
    test_coordinate_system()
    
    # Test label transformations
    test_label_transformations()
    
    # Test on real data
    print("\n" + "="*50)
    print("TESTING ON REAL DATA")
    print("="*50)
    test_sar_augmentations()
    
    print("\nTest completed!") 