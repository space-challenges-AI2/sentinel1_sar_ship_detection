#!/usr/bin/env python3
"""
HRSID JPG Dataset Denoising Script
Denoises all images in the HRSID_JPG dataset (both train and val) using FABF method
Creates a complete denoised dataset for training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import shutil

# Add the project root to path to import denoising utilities
import sys
sys.path.append(str(Path(__file__).parent))

from utils.denoising.fabf import adaptive_bilateral_filter

def denoise_image(image_path, output_path, rho=5.0, N=5, sigma=0.1, theta=None, clip=True):
    """
    Denoise a single image using FABF method
    
    Args:
        image_path: Path to input image
        output_path: Path to save denoised image
        rho: Spatial window radius (default: 5.0)
        N: Polynomial order (default: 5)
        sigma: Noise level (default: 0.1)
        theta: Target intensity (default: None, uses image itself)
        clip: Whether to clip output to [0, 1] (default: True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return False
        
        # Convert BGR to RGB (OpenCV reads as BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply denoising
        denoised = adaptive_bilateral_filter(
            img_rgb, 
            sigma_map=sigma, 
            theta_map=theta, 
            rho=rho, 
            N=N, 
            clip=clip
        )
        
        # Convert back to BGR for OpenCV saving
        denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save denoised image
        cv2.imwrite(str(output_path), denoised_bgr)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def copy_labels(input_labels_dir, output_labels_dir):
    """
    Copy label files to maintain dataset structure
    
    Args:
        input_labels_dir: Input labels directory
        output_labels_dir: Output labels directory
    
    Returns:
        int: Number of label files copied
    """
    if not input_labels_dir.exists():
        return 0
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_labels_dir.glob("*.txt"))
    for label_file in label_files:
        shutil.copy2(label_file, output_labels_dir / label_file.name)
    
    return len(label_files)

def denoise_dataset_split(input_dir, output_dir, split_name, rho=5.0, N=5, sigma=0.1, theta=None, clip=True):
    """
    Denoise images in a specific dataset split (train or val)
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for denoised images
        split_name: Name of the split (train/val)
        rho: Spatial window radius
        N: Polynomial order
        sigma: Noise level
        theta: Target intensity
        clip: Whether to clip output
    
    Returns:
        tuple: (total_images, successful_denoising, failed_denoising, labels_copied)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {split_name} split: {input_dir}")
        return 0, 0, 0, 0
    
    print(f"\nProcessing {split_name} split:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images found: {len(image_files)}")
    
    # Process images with progress bar
    successful = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc=f"Denoising {split_name} images"):
        # Create output path
        output_file = output_path / img_file.name
        
        # Denoise image
        if denoise_image(img_file, output_file, rho, N, sigma, theta, clip):
            successful += 1
        else:
            failed += 1
    
    # Copy labels if they exist
    input_labels = input_path.parent / "labels" / split_name
    output_labels = output_path.parent / "labels" / split_name
    labels_copied = copy_labels(input_labels, output_labels)
    
    return len(image_files), successful, failed, labels_copied

def denoise_full_dataset(input_dataset_dir, output_dataset_dir, rho=5.0, N=5, sigma=0.1, theta=None, clip=True):
    """
    Denoise entire HRSID dataset with both train and val splits
    
    Args:
        input_dataset_dir: Input dataset directory (e.g., HRSID_JPG)
        output_dataset_dir: Output dataset directory for denoised images
        rho: Spatial window radius
        N: Polynomial order
        sigma: Noise level
        theta: Target intensity
        clip: Whether to clip output
    
    Returns:
        dict: Summary of processing results
    """
    input_path = Path(input_dataset_dir)
    output_path = Path(output_dataset_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset directory '{input_dataset_dir}' does not exist")
    
    # Check for expected structure
    train_images_dir = input_path / "images" / "train"
    val_images_dir = input_path / "images" / "val"
    
    if not train_images_dir.exists() or not val_images_dir.exists():
        raise FileNotFoundError(f"Expected structure: {input_dataset_dir}/images/train and {input_dataset_dir}/images/val")
    
    print("=" * 80)
    print("HRSID Dataset Denoising")
    print("=" * 80)
    print(f"Input dataset: {input_dataset_dir}")
    print(f"Output dataset: {output_dataset_dir}")
    print(f"Denoising parameters: rho={rho}, N={N}, sigma={sigma}")
    print("=" * 80)
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "labels").mkdir(exist_ok=True)
    
    # Process train split
    train_output = output_path / "images" / "train"
    train_results = denoise_dataset_split(
        train_images_dir, train_output, "train", rho, N, sigma, theta, clip
    )
    
    # Process val split
    val_output = output_path / "images" / "val"
    val_results = denoise_dataset_split(
        val_images_dir, val_output, "val", rho, N, sigma, theta, clip
    )
    
    # Copy dataset YAML if it exists
    yaml_files = list(input_path.glob("*.yaml"))
    for yaml_file in yaml_files:
        shutil.copy2(yaml_file, output_path / yaml_file.name)
        print(f"Copied dataset config: {yaml_file.name}")
    
    # Summary
    total_images = train_results[0] + val_results[0]
    total_successful = train_results[1] + val_results[1]
    total_failed = train_results[2] + val_results[2]
    total_labels = train_results[3] + val_results[3]
    
    return {
        'total_images': total_images,
        'train_images': train_results[0],
        'val_images': val_results[0],
        'successful': total_successful,
        'failed': total_failed,
        'labels_copied': total_labels,
        'train_successful': train_results[1],
        'val_successful': val_results[1]
    }

def main():
    parser = argparse.ArgumentParser(description="Denoise entire HRSID JPG dataset")
    parser.add_argument("--input", "-i", 
                       default="data/HRSID_JPG",
                       help="Input dataset directory (default: data/HRSID_JPG)")
    parser.add_argument("--output", "-o", 
                       default="data/HRSID_JPG_denoised",
                       help="Output dataset directory for denoised images (default: data/HRSID_JPG_denoised)")
    parser.add_argument("--rho", type=float, default=5.0,
                       help="Spatial window radius for FABF (default: 5.0)")
    parser.add_argument("--N", type=int, default=5,
                       help="Polynomial order for FABF (default: 5)")
    parser.add_argument("--sigma", type=float, default=0.1,
                       help="Noise level for FABF (default: 0.1)")
    parser.add_argument("--theta", type=float, default=None,
                       help="Target intensity for FABF (default: None, uses image itself)")
    parser.add_argument("--no-clip", action="store_true",
                       help="Disable output clipping to [0, 1]")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.input).exists():
        print(f"Error: Input dataset directory '{args.input}' does not exist")
        return 1
    
    # Start timing
    start_time = time.time()
    
    try:
        # Denoise full dataset
        results = denoise_full_dataset(
            args.input, 
            args.output,
            rho=args.rho,
            N=args.N,
            sigma=args.sigma,
            theta=args.theta,
            clip=not args.no_clip
        )
        
        # Print results
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("Denoising Complete!")
        print("=" * 80)
        print(f"Total images processed: {results['total_images']}")
        print(f"  - Train: {results['train_images']} images")
        print(f"  - Val: {results['val_images']} images")
        print(f"Successful denoising: {results['successful']}")
        print(f"Failed denoising: {results['failed']}")
        print(f"Labels copied: {results['labels_copied']}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/results['total_images']:.2f} seconds")
        print(f"Denoised dataset saved to: {args.output}")
        print("\nDataset structure created:")
        print(f"  {args.output}/")
        print(f"  ├── images/")
        print(f"  │   ├── train/ ({results['train_successful']} denoised images)")
        print(f"  │   └── val/ ({results['val_successful']} denoised images)")
        print(f"  ├── labels/")
        print(f"  │   ├── train/ (copied)")
        print(f"  │   └── val/ (copied)")
        print(f"  └── *.yaml (dataset configs)")
        print("\nReady for training with denoised images!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())