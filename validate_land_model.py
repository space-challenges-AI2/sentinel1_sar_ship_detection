#!/usr/bin/env python3
"""
Enhanced validation script for the land model with detailed TP, FP, FN reporting.
This script demonstrates how to properly validate the land model and get comprehensive metrics.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from val import run, parse_opt

def main():
    """Main function to run enhanced land model validation"""
    
    # Parse arguments with land model defaults
    opt = parse_opt()
    
    # Override defaults for land model validation
    opt.data = ROOT / 'data/HRSID_land.yaml'
    opt.weights = ROOT / 'runs/train/experiment/weights/best.pt'
    opt.conf_thres = 0.001  # Low confidence threshold for comprehensive evaluation
    opt.iou_thres = 0.5     # Standard IoU threshold
    opt.verbose = True       # Enable verbose output
    opt.plots = True         # Enable plotting
    opt.save_txt = True      # Save detection results
    opt.save_json = True     # Save COCO format results
    
    print("="*80)
    print("LAND MODEL VALIDATION WITH ENHANCED METRICS")
    print("="*80)
    print(f"Dataset: {opt.data}")
    print(f"Weights: {opt.weights}")
    print(f"Confidence threshold: {opt.conf_thres}")
    print(f"IoU threshold: {opt.iou_thres}")
    print("="*80)
    
    # Run validation
    try:
        results = run(**vars(opt))
        print("\nValidation completed successfully!")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 