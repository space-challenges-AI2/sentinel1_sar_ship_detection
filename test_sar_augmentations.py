#!/usr/bin/env python3
"""
Test script for SAR augmentations
Visualizes augmented images with their labels to ensure proper transformations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from pathlib import Path
import sys
import os
from typing import Tuple

# Add parent directory to path to import SAR augmentations
sys.path.append(str(Path(__file__).parent))
from utils.sar_augmentations import SARAugmentations, create_synthetic_samples
import yaml

def apply_cropping(img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random cropping with label adjustment"""
    h, w = img.shape[:2]
    
    # Crop to 80% of original size
    crop_h = int(h * 0.8)
    crop_w = int(w * 0.8)
    
    # Random crop position
    y_start = random.randint(0, h - crop_h)
    x_start = random.randint(0, w - crop_w)
    
    # Crop image
    img_cropped = img[y_start:y_start + crop_h, x_start:x_start + crop_w]
    
    # Adjust labels
    if len(labels) > 0:
        labels_cropped = []
        for label in labels:
            cls, x_c, y_c, w_l, h_l = label
            
            # Convert to pixel coordinates
            x_c_pixel = x_c * w
            y_c_pixel = y_c * h
            w_pixel = w_l * w
            h_pixel = h_l * h
            
            # Adjust for crop
            x_c_new = x_c_pixel - x_start
            y_c_new = y_c_pixel - y_start
            
            # Check if label is still visible
            if (x_c_new + w_pixel/2 > 0 and x_c_new - w_pixel/2 < crop_w and
                y_c_new + h_pixel/2 > 0 and y_c_new - h_pixel/2 < crop_h):
                
                # Convert back to normalized coordinates
                x_c_norm = x_c_new / crop_w
                y_c_norm = y_c_new / crop_h
                w_norm = w_pixel / crop_w
                h_norm = h_pixel / crop_h
                
                labels_cropped.append([cls, x_c_norm, y_c_norm, w_norm, h_norm])
        
        labels_cropped = np.array(labels_cropped) if labels_cropped else np.empty((0, 5))
    else:
        labels_cropped = labels.copy()
    
    return img_cropped, labels_cropped


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
        
        # Create individual augmentation examples for better visualization
        individual_augmentations = []
        
        # 1. Horizontal Flip - simple, clean, very visible
        img_flip = cv2.flip(img.copy(), 1)  # 1 = horizontal flip
        labels_flip = labels.copy()
        if len(labels_flip) > 0:
            # Flip x-coordinates for horizontal flip
            labels_flip[:, 1] = 1.0 - labels_flip[:, 1]  # x_center = 1 - x_center
        individual_augmentations.append(("Horizontal Flip", img_flip, labels_flip))
        
        # 2. SAR Cutout - shows label filtering and realistic radar shadows
        img_cutout, labels_cutout = sar_aug.apply_sar_realistic_cutout(img.copy(), labels.copy())
        individual_augmentations.append(("SAR Cutout (Shadows)", img_cutout, labels_cutout))
        
        # Apply full SAR augmentations
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
        
        # Calculate proper subplot layout - ONLY 3 images total
        n_cols = 3  # Original + 2 augmentations = 3 total
        
        # Create subplot layout
        fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 12))
        fig.suptitle(f'SAR Augmentation Test - {img_path.name}', fontsize=16)
        
        # Original image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_with_boxes = draw_boxes(img_rgb, labels, color=(0, 255, 0))
        axes[0, 0].imshow(img_with_boxes)
        axes[0, 0].set_title(f'Original ({len(labels)} ships)')
        axes[0, 0].axis('off')
        
        # Individual augmentations - ONLY 2 augmentations
        for j, (aug_name, aug_img, aug_labels) in enumerate(individual_augmentations):
            aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            aug_img_with_boxes = draw_boxes(aug_img_rgb, aug_labels, color=(255, 165, 0))  # Orange
            axes[0, 1+j].imshow(aug_img_with_boxes)
            axes[0, 1+j].set_title(f'{aug_name}\n({len(aug_labels)} ships)')
            axes[0, 1+j].axis('off')
        
        # Bottom row: Statistics and comparisons
        # Original labels
        axes[1, 0].axis('off')
        if len(labels) > 0:
            axes[1, 0].text(0.1, 0.8, f'Original Labels:\n{labels}', 
                           transform=axes[1, 0].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        # Augmentation statistics
        axes[1, 1].axis('off')
        stats_text = f"""Main Augmentation Probabilities:
        • Horizontal Flip: Built-in YOLOv5
        • SAR Cutout (Shadows): {hyp.get('cutout_prob', 0.3):.1%}
        
        Additional (not shown):
        • SAR Speckle Noise: {hyp.get('noise_prob', 0.2):.1%}
        • Gaussian Blur: {hyp.get('blur_prob', 0.15):.1%}
        • Unsharp Mask: {hyp.get('sharpen_prob', 0.15):.1%}"""
        
        axes[1, 1].text(0.1, 0.8, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        # Combined label comparison and YOLOv5 info
        axes[1, 2].axis('off')
        combined_text = f"""Label Changes:
        • Original: {len(labels)} ships
        • After Flip: {len(labels_flip)} ships (x-coords flipped)
        • After Cutout: {len(labels_cutout)} ships
        • Final Augmented: {len(labels_aug)} ships

        YOLOv5 Built-in:
        • Mosaic: {hyp.get('mosaic', 0.8):.1%}
        • MixUp: {hyp.get('mixup', 0.2):.1%}
        • Copy-Paste: {hyp.get('copy_paste', 0.1):.1%}
        • Rotation: ±{hyp.get('degrees', 15.0)}°
        • Scale: ±{hyp.get('scale', 0.3):.1%}"""
        
        axes[1, 2].text(0.1, 0.8, combined_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        # All 3 bottom row subplots are used, no need to hide any
        
        plt.tight_layout()
        plt.show()
        
        # Save test results
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save individual augmentation examples
        for aug_name, aug_img, aug_labels in individual_augmentations:
            # Clean name for filename
            clean_name = aug_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            cv2.imwrite(str(output_dir / f"{clean_name}_{img_path.name}"), aug_img)
            
            if len(aug_labels) > 0:
                np.savetxt(str(output_dir / f"{clean_name}_{img_path.stem}.txt"), aug_labels, fmt='%.6f')
        
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

def test_fabf_denoising():
    """Test FABF denoising on sample images"""
    print("\n" + "="*50)
    print("TESTING FABF DENOISING")
    print("="*50)
    
    try:
        from utils.denoising.fabf import adaptive_bilateral_filter
        print("FABF denoising module imported successfully")
        
        # Create a test image with synthetic noise
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add synthetic SAR-like noise
        noise_factor = 0.1
        speckle_noise = np.random.exponential(1.0, test_img.shape)
        test_img_noisy = test_img.astype(np.float32) * (1 + noise_factor * (speckle_noise - 1))
        test_img_noisy = np.clip(test_img_noisy, 0, 255).astype(np.uint8)
        
        print(f"Test image shape: {test_img.shape}")
        print(f"Added synthetic SAR noise with factor: {noise_factor}")
        
        # Apply FABF denoising
        try:
            # Try to apply FABF denoising
            img_denoised = adaptive_bilateral_filter(test_img_noisy, sigma_map=0.115, rho=3.0, N=1)
            print("FABF denoising applied successfully")
            
            # Save test images
            output_dir = Path("test_results")
            output_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_dir / "fabf_original.jpg"), test_img)
            cv2.imwrite(str(output_dir / "fabf_noisy.jpg"), test_img_noisy)
            cv2.imwrite(str(output_dir / "fabf_denoised.jpg"), img_denoised)
            
            print("FABF denoising test images saved to test_results/")
            
            return True
            
        except Exception as e:
            print(f"Error applying FABF denoising: {e}")
            return False
            
    except ImportError as e:
        print(f"Could not import FABF denoising module: {e}")
        return False

def summarize_hrsid_augmentations():
    """Summarize all augmentations used in HRSID augmented dataset creation"""
    print("\n" + "="*80)
    print("HRSID AUGMENTED DATASET - COMPLETE AUGMENTATION SUMMARY")
    print("="*80)
    
    print("\n📊 DATASET TRANSFORMATION:")
    print("   • Original HRSID_land_denoised: ~393 images")
    print("   • Target: ~2000+ images (5x+ increase)")
    print("   • Strategy: Aggressive augmentation + synthetic samples")
    
    print("\n🔧 SAR-SPECIFIC AUGMENTATIONS (Custom Implementation):")
    print("   • Speckle Noise Addition (20% prob): Multiplicative exponential + Gaussian noise")
    print("   • Gaussian Blur (15% prob): Atmospheric effects simulation (3x3, 5x5, 7x7 kernels)")
    print("   • Unsharp Masking (15% prob): Edge enhancement for SAR imagery")
    print("   • SAR-Realistic Cutout (30% prob): Dark radar shadows, label filtering (>60% overlap)")
    print("   • Elastic Deformation (10% prob): Terrain variation simulation with label transformation")
    
    print("\n🎯 TOP 2 AUGMENTATIONS (Shown in Test):")
    print("   1. Horizontal Flip: Simple, clean, very visible geometric transformation")
    print("   2. SAR Cutout (Shadows): Shows label filtering and realistic radar shadows")
    
    print("\n🔄 SYNTHETIC SAMPLE GENERATION:")
    print("   • Horizontal Flip: Mirror image with x-coordinate flipping")
    print("   • Vertical Flip: Vertical mirror with y-coordinate flipping")
    print("   • Small Rotation: ±10° with proper label transformation")
    
    print("\n🚀 YOLOv5 BUILT-IN AUGMENTATIONS:")
    print("   • Mosaic (80% prob): Multi-image combination")
    print("   • MixUp (20% prob): Image blending")
    print("   • Copy-Paste (10% prob): Object transplantation")
    print("   • Geometric: Rotation (±15°), Translation (±20%), Scale (±30%), Shear (±5°)")
    print("   • Flip: Horizontal (50%), Vertical (30%)")
    
    print("\n🧹 DENOISING PREPROCESSING:")
    print("   • Method: FABF (Fast Adaptive Bilateral Filter)")
    print("   • Parameters: σ=0.115, ρ=3.0, N=1")
    print("   • Purpose: Remove SAR speckle noise before augmentation")
    
    print("\n📈 AUGMENTATION EFFECTS ON LABELS:")
    print("   • Non-affecting: Noise, blur, sharpen (labels unchanged)")
    print("   • Label-filtering: Cutout removes heavily obscured objects")
    print("   • Label-transforming: Elastic deformation, flips, rotation")
    print("   • Coordinate system: Normalized (0-1) with proper bounds checking")
    
    print("\n🎯 DESIGN PRINCIPLES:")
    print("   • SAR-specific: Tailored for radar imagery characteristics")
    print("   • Label-preserving: Maintains annotation integrity")
    print("   • Realistic: Simulates real-world SAR imaging conditions")
    print("   • Conservative: Avoids over-augmentation that could hurt performance")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("SAR Augmentation Test Suite")
    print("="*50)
    
    # Show complete augmentation summary
    summarize_hrsid_augmentations()
    
    # Test FABF denoising first
    fabf_success = test_fabf_denoising()
    
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