#!/usr/bin/env python3
"""
Training script for HRSID_augmented dataset
Combines original ship images with augmented land-only images
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=-1, help='batch size')
    parser.add_argument('--weights', type=str, default='', help='initial weights')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    args = parser.parse_args()
    
    # Dataset path - use the augmented dataset
    data_yaml = "data/HRSID_augmented/dataset.yaml"
    
    # Hyperparameters for augmented training
    hyp_yaml = "data/hyp/hyp.land_denoised_augmented.yaml"
    
    # Build the command dynamically to handle empty weights
    cmd_parts = [
        "python train.py",
        f"--cfg models/yolov5n.yaml",
        f"--data {data_yaml}",
        f"--hyp {hyp_yaml}",
        f"--epochs {args.epochs}"
    ]
    
    # Only add --weights if it's not empty
    if args.weights:
        cmd_parts.append(f"--weights {args.weights}")
    
    # Join the command parts
    cmd = " \\\n        ".join(cmd_parts)
    
    print("Running training command:")
    print(cmd)
    
    # Execute training
    import subprocess
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main() 