#!/usr/bin/env python3
"""
Training script for HRSID_land_denoised dataset with conservative SAR augmentations
This script uses gentle augmentations to improve performance while maintaining data quality
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train import train, parse_opt
from utils.general import LOGGER, colorstr

def main():
    """Main training function with conservative augmentations"""
    
    # Parse arguments
    opt = parse_opt()
    
    # Override with dataset-specific settings
    opt.data = 'data/HRSID_land_denoised.yaml'
    opt.hyp = 'data/hyp/hyp.land_denoised_conservative.yaml'  # Use conservative hyperparameters
    opt.epochs = 350  # More epochs for small dataset
    opt.batch_size = -1  # Autobatch mode - automatically determines optimal batch size
    opt.imgsz = 640
    opt.weights = ''  # Start with pretrained weights
    opt.project = 'runs/train_land_denoised_conservative'
    opt.name = 'exp'
    
    # Print configuration
    LOGGER.info(colorstr('Training Configuration (Conservative Augmentations):'))
    LOGGER.info(f'Dataset: {opt.data}')
    LOGGER.info(f'Hyperparameters: {opt.hyp}')
    LOGGER.info(f'Epochs: {opt.epochs}')
    LOGGER.info(f'Batch size: {opt.batch_size}')
    LOGGER.info(f'Image size: {opt.imgsz}')
    LOGGER.info(f'Weights: {opt.weights}')
    LOGGER.info(f'Project: {opt.project}')
    
    # Check if dataset exists
    if not Path(opt.data).exists():
        LOGGER.error(f'Dataset file not found: {opt.data}')
        return
    
    if not Path(opt.hyp).exists():
        LOGGER.error(f'Hyperparameters file not found: {opt.hyp}')
        return
    
    # Check if weights exist
    if not Path(opt.weights).exists() and not opt.weights.startswith('yolov5'):
        LOGGER.error(f'Weights file not found: {opt.weights}')
        return
    
    # Import and run training
    try:
        from utils.general import init_seeds
        from utils.torch_utils import select_device
        from utils.callbacks import Callbacks
        from utils.loggers import Loggers
        from utils.general import methods
        
        # Initialize
        init_seeds(1)
        device = select_device(opt.device, batch_size=opt.batch_size)
        
        # Setup logging - FOLLOW THE EXACT PATTERN FROM MAIN TRAINING SCRIPT
        # Construct save_dir from project and name
        from utils.general import increment_path
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=True)
        
        # Create callbacks and loggers following the main training script pattern
        callbacks = Callbacks()
        loggers = Loggers(opt.save_dir, opt.weights, opt, opt.hyp, LOGGER)
        
        # Register corresponding operations
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        
        # Load hyperparameters
        import yaml
        with open(opt.hyp, 'r') as f:
            hyp = yaml.safe_load(f)
        
        LOGGER.info(colorstr('Starting training with conservative SAR augmentations...'))
        LOGGER.info(colorstr('This should improve recall and precision by using gentler augmentations'))
        
        # Run training
        train(hyp, opt, device, callbacks)
        
        LOGGER.info(colorstr('Training completed successfully!'))
        
    except Exception as e:
        LOGGER.error(f'Training failed: {e}')
        raise

if __name__ == "__main__":
    main() 