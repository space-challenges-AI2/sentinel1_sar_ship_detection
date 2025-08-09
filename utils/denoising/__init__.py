# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Denoising utilities for YOLOv5 Ship Detection from SAR Images
Author: @amanarora9848 (Aman Arora)
"""

from .base import (
    DenoisingMethod,
    DenoisingRegistry,
    apply_denoising,
    ensure_float01,
    validate_image
)

# Try to import and register the FABF method
try:
    from .fabf import adaptive_bilateral_filter
    
    class FABFDenoiser(DenoisingMethod):
        """
        Fast Adaptive Bilateral Filter (FABF) denoiser.
        This is a wrapper around the adaptive_bilateral_filter function that implements the DenoisingMethod interface.
        """
        
        def __init__(self, rho: float = 5.0, N: int = 5, sigma_map=None, theta_map=None, clip: bool = True):
            """
            Initialize the FABF denoiser.
            
            Args:
                rho: Spatial window radius (default: 5.0)
                N: polynomial order for approximation (default: 5)
                sigma_map: Noise level map for adaptive filtering
                theta_map: target intensity map
                clip: whether to clip the output to [0, 1]
            """
            super().__init__(rho=rho, N=N, sigma_map=sigma_map, theta_map=theta_map, clip=clip)
        
        def __call__(self, image, **kwargs):
            """
            Apply FABF denoising to the image.
            
            Args:
                image: Input image as numpy array
                **kwargs: Additional parameters (overrides constructor params)
            
            Returns:
                Denoised image
            """
            # Merge constructor params with call params
            params = self.get_params().copy()
            params.update(kwargs)
            
            return adaptive_bilateral_filter(
                image,
                sigma_map=params.get('sigma_map'),
                theta_map=params.get('theta_map'),
                rho=params.get('rho', 5.0),
                N=params.get('N', 5),
                clip=params.get('clip', True)
            )
        
        def get_name(self) -> str:
            """Return the method name."""
            return 'fabf'
    
    # Register the FABF method
    DenoisingRegistry.register('fabf', FABFDenoiser)
    
    # Export all functions and classes
    __all__ = [
        'DenoisingMethod',
        'DenoisingRegistry',
        'apply_denoising',
        'ensure_float01',
        'validate_image',
        'adaptive_bilateral_filter',
        'FABFDenoiser'
    ]
    
except ImportError:
    # If FABF is not available, only export the base classes
    __all__ = [
        'DenoisingMethod',
        'DenoisingRegistry',
        'apply_denoising',
        'ensure_float01',
        'validate_image'
    ]