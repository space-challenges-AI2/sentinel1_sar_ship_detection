import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from utils.sar_augmentations import create_sar_augmentation_pipeline, create_synthetic_samples
import yaml

def create_augmented_dataset():
    """
    We create HRSID_augmented dataset by combining:
    1. Original HRSID_JPG images (with ships)
    2. Augmented HRSID_land_main images (land-only, augmented)
    """
    
    # Paths
    hrsid_jpg_path = Path("data/HRSID_JPG")
    hrsid_land_path = Path("data/HRSID_land_main")
    output_path = Path("data/HRSID_augmented")
    
    # Create output directories
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Load hyperparameters for augmentation
    with open("data/hyp/hyp.land_denoised_augmented.yaml", 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Initialize SAR augmentation pipeline
    sar_aug = create_sar_augmentation_pipeline(hyp)
    
    # Step 1: Copy original HRSID_JPG images (with ships)
    print("Copying original HRSID_JPG images...")
    copy_dataset(hrsid_jpg_path, output_path, "original")
    
    # Step 2: Create augmented versions of land-only images
    print("Creating augmented land-only images...")
    create_augmented_land_images(hrsid_land_path, output_path, sar_aug, hyp)
    
    # Step 3: Create dataset configuration
    create_dataset_config(output_path)  
    
    print(f"Dataset created successfully at {output_path}")
    print("Original images + augmented land images = enhanced dataset")

def copy_dataset(source_path, dest_path, prefix=""):
    """Copy images and labels from source to destination"""
    
    # Copy training images
    train_images = list((source_path / "images" / "train").glob("*.jpg"))
    for img_path in train_images:
        dest_img = dest_path / "images" / "train" / f"{prefix}_{img_path.name}"
        shutil.copy2(img_path, dest_img)
        
        # Copy corresponding label if it exists
        label_path = source_path / "labels" / "train" / f"{img_path.stem}.txt"
        if label_path.exists():
            dest_label = dest_path / "labels" / "train" / f"{prefix}_{label_path.name}"
            shutil.copy2(label_path, dest_label)
    
    # Copy validation images
    val_images = list((source_path / "images" / "val").glob("*.jpg"))
    for img_path in val_images:
        dest_img = dest_path / "images" / "val" / f"{prefix}_{img_path.name}"
        shutil.copy2(img_path, dest_img)
        
        # Copy corresponding label if it exists
        label_path = source_path / "labels" / "val" / f"{img_path.stem}.txt"
        if label_path.exists():
            dest_label = dest_path / "labels" / "val" / f"{prefix}_{label_path.name}"
            shutil.copy2(label_path, dest_label)

def create_augmented_land_images(land_path, output_path, sar_aug, hyp):
    """Create augmented versions of land-only images"""
    
    # Get all land images
    train_images = list((land_path / "images" / "train").glob("*.jpg"))
    
    # Number of augmented versions per image
    n_augmentations = 3  # Adjust
    
    for i, img_path in enumerate(train_images):
        print(f"Processing {img_path.name} ({i+1}/{len(train_images)})")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Load labels (if any)
        label_path = land_path / "labels" / "train" / f"{img_path.stem}.txt"
        labels = np.empty((0, 5))
        if label_path.exists():
            labels = np.loadtxt(str(label_path))
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
        
        # Create augmented versions
        for aug_idx in range(n_augmentations):
            # Apply SAR augmentations
            img_aug, labels_aug = sar_aug(img, labels)
            
            # Create additional synthetic samples
            synthetic_samples = create_synthetic_samples(img_aug, labels_aug, n_samples=1)
            
            for syn_idx, (syn_img, syn_labels) in enumerate(synthetic_samples):
                # Save augmented image
                aug_name = f"aug_{img_path.stem}_v{aug_idx}_s{syn_idx}.jpg"
                aug_img_path = output_path / "images" / "train" / aug_name
                cv2.imwrite(str(aug_img_path), syn_img)
                
                # Save augmented labels
                if len(syn_labels) > 0:
                    aug_label_path = output_path / "labels" / "train" / f"{aug_name[:-4]}.txt"
                    np.savetxt(str(aug_label_path), syn_labels, fmt='%.6f')
                else:
                    # Create empty label file for land-only images
                    aug_label_path = output_path / "labels" / "train" / f"{aug_name[:-4]}.txt"
                    aug_label_path.touch()

def create_dataset_config(output_path):
    """Create dataset configuration file"""
    
    # Count images
    train_count = len(list((output_path / "images" / "train").glob("*.jpg")))
    val_count = len(list((output_path / "images" / "val").glob("*.jpg")))
    
    config = {
        'path': str(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'nc': 1,
        'names': ['ship'],
        'train_count': train_count,
        'val_count': val_count
    }
    
    # Save config
    config_path = output_path / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config saved to {config_path}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

if __name__ == "__main__":
    create_augmented_dataset() 