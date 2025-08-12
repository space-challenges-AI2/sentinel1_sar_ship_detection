"""
Integration utilities for denoising in YOLOv5 pipeline
Author: @amanarora9848 (Aman Arora)
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Union
from .base import apply_denoising

def prepare_denoise_params_from_args(args) -> Dict[str, Any]:
    """
    Prepare denoising parameters from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary of denoising parameters
    """
    return {
        'enabled': getattr(args, 'denoise', 0.0) > 0.0,
        'probability': getattr(args, 'denoise', 0.0),
        'method': getattr(args, 'denoise_method', 'fabf'),
        'rho': getattr(args, 'denoise_rho', 5.0),
        'N': getattr(args, 'denoise_N', 5),
        'sigma': getattr(args, 'denoise_sigma', 0.1),
        'theta': getattr(args, 'denoise_theta', None),
        'clip': getattr(args, 'denoise_clip', True)
    }

def prepare_denoise_params_from_hyp(hyp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare denoising parameters from hyperparameters dictionary.
    
    Args:
        hyp: Hyperparameters dictionary
        
    Returns:
        Dictionary of denoising parameters
    """
    return {
        'enabled': hyp.get('denoise', 0.0) > 0.0,
        'probability': hyp.get('denoise', 0.0),
        'method': hyp.get('denoise_method', 'fabf'),
        'rho': hyp.get('denoise_rho', 5.0),
        'N': hyp.get('denoise_N', 5),
        'sigma': hyp.get('denoise_sigma', 0.1),
        'theta': None if hyp.get('denoise_theta') == 'null' else hyp.get('denoise_theta'),
        'clip': hyp.get('denoise_clip', True)
    }

def apply_denoising_to_image(image: np.ndarray, 
                           denoise_params: Dict[str, Any],
                           convert_bgr: bool = True) -> np.ndarray:
    """
    Apply denoising to an image with proper color space handling.
    
    Args:
        image: Input image (BGR if convert_bgr=True, RGB otherwise)
        denoise_params: Denoising parameters dictionary
        convert_bgr: Whether to convert BGR to RGB before denoising
        
    Returns:
        Denoised image in the same color space as input
    """
    if not denoise_params.get('enabled', False):
        return image
    
    # Convert BGR to RGB if needed
    if convert_bgr and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Handle theta parameter properly
    theta = denoise_params.get('theta')
    if theta == 'null' or theta == 'None':
        theta = None
    
    # Apply denoising
    denoised_rgb = apply_denoising(
        image_rgb,
        method=denoise_params.get('method', 'fabf'),
        probability=denoise_params.get('probability', 1.0),
        rho=denoise_params.get('rho', 5.0),
        N=denoise_params.get('N', 5),
        sigma_map=denoise_params.get('sigma', 0.1),
        theta_map=theta,  # Use cleaned theta value
        clip=denoise_params.get('clip', True)
    )
    
    # Convert back to BGR if needed
    if convert_bgr and image.shape[2] == 3:
        return cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)
    else:
        return denoised_rgb

def log_denoising_config(denoise_params: Dict[str, Any], logger=None):
    """
    Log denoising configuration for debugging and tracking.
    
    Args:
        denoise_params: Denoising parameters dictionary
        logger: Logger instance (optional)
    """
    if not denoise_params.get('enabled', False):
        return
    
    config_str = f"Denoising: {denoise_params['method']} (p={denoise_params['probability']:.2f})"
    if denoise_params['method'] == 'fabf':
        config_str += f", rho={denoise_params['rho']}, N={denoise_params['N']}"
        config_str += f", sigma={denoise_params['sigma']}"
        if denoise_params['theta'] is not None:
            config_str += f", theta={denoise_params['theta']}"
        config_str += f", clip={denoise_params['clip']}"
    
    if logger:
        logger.info(config_str)
    else:
        print(config_str)