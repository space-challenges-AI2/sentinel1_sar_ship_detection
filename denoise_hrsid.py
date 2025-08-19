#!/usr/bin/env python3
"""
HRSID Land Dataset Denoising Script
Denoises all images in the HRSID land dataset (both train and val) using FABF method
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
import yaml

# Add the project root to path to import denoising utilities
import sys
sys.path.append(str(Path(__file__).parent))

from utils.denoising.fabf import adaptive_bilateral_filter

def load_denoising_params(hyp_file_path="data/hyp/hyp.scratch-low.yaml"):
    """
    Load denoising parameters from hyperparameter file
    
    Args:
        hyp_file_path: Path to hyperparameter YAML file
    
    Returns:
        dict: Denoising parameters
    """
    try:
        with open(hyp_file_path, 'r') as f:
            hyp_data = yaml.safe_load(f)
        
        # Extract denoising parameters
        denoise_params = {
            'rho': hyp_data.get('denoise_rho', 3.0),
            'N': hyp_data.get('denoise_N', 1),
            'sigma': hyp_data.get('denoise_sigma', 0.115),
            'theta': hyp_data.get('denoise_theta', None),
            'clip': hyp_data.get('denoise_clip', True)
        }
        
        # Convert 'null' string to None for theta
        if denoise_params['theta'] == 'null':
            denoise_params['theta'] = None
        
        print(f"Loaded denoising parameters from {hyp_file_path}:")
        print(f"  rho: {denoise_params['rho']}")
        print(f"  N: {denoise_params['N']}")
        print(f"  sigma: {denoise_params['sigma']}")
        print(f"  theta: {denoise_params['theta']}")
        print(f"  clip: {denoise_params['clip']}")
        
        return denoise_params
        
    except Exception as e:
        print(f"Warning: Could not load hyperparameter file {hyp_file_path}: {e}")
        print("Using default denoising parameters")
        return {
            'rho': 3.0,
            'N': 1,
            'sigma': 0.115,
            'theta': None,
            'clip': True
        }

def denoise_image(image_path, output_path, rho=3.0, N=1, sigma=0.115, theta=None, clip=True):
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
        print(f"Warning: Labels directory not found: {input_labels_dir}")
        return 0
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_labels_dir.glob("*.txt"))
    if not label_files:
        print(f"Warning: No .txt label files found in {input_labels_dir}")
        return 0
    
    print(f"Copying {len(label_files)} label files from {input_labels_dir} to {output_labels_dir}")
    
    for label_file in tqdm(label_files, desc="Copying labels"):
        try:
            shutil.copy2(label_file, output_labels_dir / label_file.name)
        except Exception as e:
            print(f"Warning: Failed to copy {label_file}: {e}")
    
    return len(label_files)

def denoise_dataset_split(input_dir, output_dir, split_name, rho=3.0, N=1, sigma=0.115, theta=None, clip=True):
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
    
    # Sort files for consistent processing order
    image_files.sort(key=lambda x: x.name)
    
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
    # The labels are at the dataset root level, not at the images level
    input_labels = input_path.parent.parent / "labels" / split_name
    output_labels = output_path.parent.parent / "labels" / split_name
    labels_copied = copy_labels(input_labels, output_labels)
    
    return len(image_files), successful, failed, labels_copied

def denoise_full_dataset(input_dataset_dir, output_dataset_dir, rho=3.0, N=1, sigma=0.115, theta=None, clip=True):
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
    train_labels_dir = input_path / "labels" / "train"
    val_labels_dir = input_path / "labels" / "val"
    
    if not train_images_dir.exists() or not val_images_dir.exists():
        raise FileNotFoundError(f"Expected structure: {input_dataset_dir}/images/train and {input_dataset_dir}/images/val")
    
    if not train_labels_dir.exists() or not val_labels_dir.exists():
        print(f"Warning: Labels directories not found. Expected: {input_dataset_dir}/labels/train and {input_dataset_dir}/labels/val")
        print("This may result in a dataset without labels!")
    
    # Verify we have images and labels
    train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.jpeg")) + list(train_images_dir.glob("*.png"))
    val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.jpeg")) + list(val_images_dir.glob("*.png"))
    train_labels = list(train_labels_dir.glob("*.txt"))
    val_labels = list(val_labels_dir.glob("*.txt"))
    
    print(f"Found {len(train_images)} training images and {len(train_labels)} training labels")
    print(f"Found {len(val_images)} validation images and {len(val_labels)} validation labels")
    
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

def denoise_dataset_flexible(input_dataset_dir, output_dataset_dir, rho=3.0, N=1, sigma=0.115, theta=None, clip=True):
    """
    Denoise dataset with flexible structure detection
    
    Args:
        input_dataset_dir: Input dataset directory
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
    
    # Detect dataset structure
    dataset_structure = detect_dataset_structure(input_path)
    print(f"Detected dataset structure: {dataset_structure['type']}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_structure['type'] == 'standard_hrsid':
        # Standard HRSID structure with train/val splits
        return denoise_full_dataset(input_path, output_path, rho, N, sigma, theta, clip)
    
    elif dataset_structure['type'] == 'flat_images':
        # Flat structure with images directly in dataset root
        return denoise_flat_dataset(input_path, output_path, rho, N, sigma, theta, clip)
    
    elif dataset_structure['type'] == 'mixed':
        # Mixed structure - handle both cases
        return denoise_mixed_dataset(input_path, output_path, rho, N, sigma, theta, clip)
    
    else:
        raise ValueError(f"Unknown dataset structure: {dataset_structure}")

def detect_dataset_structure(dataset_path):
    """
    Detect the structure of the dataset
    
    Returns:
        dict: Structure information
    """
    # Check for standard HRSID structure
    train_images_dir = dataset_path / "images" / "train"
    val_images_dir = dataset_path / "images" / "val"
    
    # Check for flat structure (images directly in dataset)
    flat_images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.jpeg")) + list(dataset_path.glob("*.png"))
    
    # Check for subdirectories with images
    subdir_images = []
    for subdir in dataset_path.iterdir():
        if subdir.is_dir():
            subdir_images.extend(list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png")))
    
    if train_images_dir.exists() and val_images_dir.exists():
        return {
            'type': 'standard_hrsid',
            'train_images': len(list(train_images_dir.glob("*.jpg"))),
            'val_images': len(list(val_images_dir.glob("*.jpg")))
        }
    elif len(flat_images) > 0:
        return {
            'type': 'flat_images',
            'total_images': len(flat_images),
            'image_files': flat_images
        }
    elif len(subdir_images) > 0:
        return {
            'type': 'mixed',
            'subdir_images': subdir_images,
            'subdirs': [d for d in dataset_path.iterdir() if d.is_dir()]
        }
    else:
        return {
            'type': 'unknown',
            'message': 'No recognizable dataset structure found'
        }

def denoise_flat_dataset(input_path, output_path, rho, N, sigma, theta, clip):
    """
    Denoise dataset with flat structure (images directly in dataset root)
    """
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_path}")
        return {'total_images': 0, 'successful': 0, 'failed': 0, 'labels_copied': 0}
    
    print(f"\nProcessing flat dataset:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Images found: {len(image_files)}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images with progress bar
    successful = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc="Denoising images"):
        # Create output path
        output_file = output_path / img_file.name
        
        # Denoise image
        if denoise_image(img_file, output_file, rho, N, sigma, theta, clip):
            successful += 1
        else:
            failed += 1
    
    # Copy labels if they exist in the same directory
    labels_copied = 0
    label_files = list(input_path.glob("*.txt"))
    if label_files:
        for label_file in label_files:
            shutil.copy2(label_file, output_path / label_file.name)
        labels_copied = len(label_files)
        print(f"Copied {labels_copied} label files")
    
    # Copy any YAML config files
    yaml_files = list(input_path.glob("*.yaml"))
    for yaml_file in yaml_files:
        shutil.copy2(yaml_file, output_path / yaml_file.name)
        print(f"Copied dataset config: {yaml_file.name}")
    
    return {
        'total_images': len(image_files),
        'successful': successful,
        'failed': failed,
        'labels_copied': labels_copied
    }

def denoise_mixed_dataset(input_path, output_path, rho, N, sigma, theta, clip):
    """
    Denoise dataset with mixed structure (images in subdirectories)
    """
    # Find all images in subdirectories
    all_images = []
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            subdir_images = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
            all_images.extend([(subdir, img) for img in subdir_images])
    
    if not all_images:
        print(f"No image files found in subdirectories of {input_path}")
        return {'total_images': 0, 'successful': 0, 'failed': 0, 'labels_copied': 0}
    
    print(f"\nProcessing mixed dataset:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Images found: {len(all_images)}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images with progress bar
    successful = 0
    failed = 0
    
    for subdir, img_file in tqdm(all_images, desc="Denoising images"):
        # Create output subdirectory
        output_subdir = output_path / subdir.name
        output_subdir.mkdir(exist_ok=True)
        
        # Create output path
        output_file = output_subdir / img_file.name
        
        # Denoise image
        if denoise_image(img_file, output_file, rho, N, sigma, theta, clip):
            successful += 1
        else:
            failed += 1
        
        # Copy labels if they exist
        label_file = subdir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, output_subdir / label_file.name)
    
    return {
        'total_images': len(all_images),
        'successful': successful,
        'failed': failed,
        'labels_copied': 0  # Labels are copied per image
    }

def main():
    parser = argparse.ArgumentParser(description="Denoise HRSID land dataset")
    parser.add_argument("--input", "-i", 
                       default="data/HRSID_land_main",
                       help="Input dataset directory (default: data/HRSID_land_main)")
    parser.add_argument("--output", "-o", 
                       default="data/HRSID_land_denoised",
                       help="Output dataset directory for denoised images (default: data/HRSID_land_denoised)")
    parser.add_argument("--single-image", "-s",
                       help="Denoise a single image instead of full dataset (provide image path)")
    parser.add_argument("--single-output", "-so",
                       help="Output path for single image denoising (required with --single-image)")
    parser.add_argument("--hyp", 
                       default="data/hyp/hyp.scratch-low.yaml",
                       help="Hyperparameter file path (default: data/hyp/hyp.scratch-low.yaml)")
    parser.add_argument("--rho", type=float, default=None,
                       help="Spatial window radius for FABF (overrides hyp file)")
    parser.add_argument("--N", type=int, default=None,
                       help="Polynomial order for FABF (overrides hyp file)")
    parser.add_argument("--sigma", type=float, default=None,
                       help="Noise level for FABF (overrides hyp file)")
    parser.add_argument("--theta", type=float, default=None,
                       help="Target intensity for FABF (overrides hyp file)")
    parser.add_argument("--no-clip", action="store_true",
                       help="Disable output clipping to [0, 1] (overrides hyp file)")
    
    args = parser.parse_args()
    
    # Load denoising parameters from hyperparameter file
    denoise_params = load_denoising_params(args.hyp)
    
    # Override with command line arguments if provided
    if args.rho is not None:
        denoise_params['rho'] = args.rho
    if args.N is not None:
        denoise_params['N'] = args.N
    if args.sigma is not None:
        denoise_params['sigma'] = args.sigma
    if args.theta is not None:
        denoise_params['theta'] = args.theta
    if args.no_clip:
        denoise_params['clip'] = False
    
    # Handle single image denoising
    if args.single_image:
        if not args.single_output:
            print("Error: --single-output is required when using --single-image")
            return 1
        
        input_image = Path(args.single_image)
        output_image = Path(args.single_output)
        
        if not input_image.exists():
            print(f"Error: Input image '{args.single_image}' does not exist")
            return 1
        
        print("=" * 80)
        print("Single Image Denoising")
        print("=" * 80)
        print(f"Input image: {args.single_image}")
        print(f"Output image: {args.single_output}")
        print(f"Denoising parameters: rho={denoise_params['rho']}, N={denoise_params['N']}, sigma={denoise_params['sigma']}")
        print("=" * 80)
        
        # Create output directory if it doesn't exist
        output_image.parent.mkdir(parents=True, exist_ok=True)
        
        # Denoise single image
        start_time = time.time()
        success = denoise_image(
            image_path=input_image,
            output_path=output_image,
            rho=denoise_params['rho'],
            N=denoise_params['N'],
            sigma=denoise_params['sigma'],
            theta=denoise_params['theta'],
            clip=denoise_params['clip']
        )
        
        if success:
            elapsed_time = time.time() - start_time
            print(f"\n✅ Successfully denoised image in {elapsed_time:.2f} seconds")
            print(f"Output saved to: {args.single_output}")
        else:
            print("\n❌ Failed to denoise image")
            return 1
        
        return 0
    
    # Validate input directory for full dataset processing
    if not Path(args.input).exists():
        print(f"Error: Input dataset directory '{args.input}' does not exist")
        return 1
    
    # Start timing
    start_time = time.time()
    
    try:
        # Use flexible dataset denoising
        results = denoise_dataset_flexible(
            args.input, 
            args.output,
            rho=denoise_params['rho'],
            N=denoise_params['N'],
            sigma=denoise_params['sigma'],
            theta=denoise_params['theta'],
            clip=denoise_params['clip']
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
        print(f"  │   └── val/ ({results['train_successful']} denoised images)")
        print(f"  ├── labels/")
        print(f"  │   ├── train/ (copied)")
        print(f"  │   └── val/ (copied)")
        print(f"  └── *.yaml (dataset configs)")
        
        # Verify the final dataset structure
        print("\nVerifying final dataset structure...")
        verify_dataset_structure(args.output)
        
        print("\nReady for training with denoised images!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

def verify_dataset_structure(dataset_dir):
    """
    Verify that the denoised dataset has the correct structure and files
    
    Args:
        dataset_dir: Path to the denoised dataset directory
    """
    dataset_path = Path(dataset_dir)
    
    # Check main directories
    required_dirs = ["images", "labels"]
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"❌ Missing required directory: {dir_name}")
            return False
    
    # Check image splits
    image_splits = ["train", "val"]
    for split in image_splits:
        image_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        
        if not image_dir.exists():
            print(f"❌ Missing image directory: images/{split}")
            return False
        
        if not label_dir.exists():
            print(f"❌ Missing label directory: labels/{split}")
            return False
        
        # Count files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
        label_files = list(label_dir.glob("*.txt"))
        
        print(f"✅ {split}: {len(image_files)} images, {len(label_files)} labels")
        
        # Check if we have corresponding labels for images
        if len(image_files) > 0 and len(label_files) == 0:
            print(f"⚠️  Warning: {split} has images but no labels!")
        elif len(image_files) != len(label_files):
            print(f"⚠️  Warning: {split} has {len(image_files)} images but {len(label_files)} labels")
    
    # Check for YAML config
    yaml_files = list(dataset_path.glob("*.yaml"))
    if yaml_files:
        print(f"✅ Found {len(yaml_files)} dataset configuration file(s)")
        for yaml_file in yaml_files:
            print(f"   - {yaml_file.name}")
    else:
        print("⚠️  Warning: No dataset configuration files found")
    
    print("✅ Dataset structure verification complete!")
    return True

if __name__ == "__main__":
    exit(main())