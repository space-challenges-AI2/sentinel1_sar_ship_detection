"""
Denoising base classes and interfaces
Author: @amanarora9848 (Aman Arora)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import torch

class DenoisingMethod(ABC):
    """
    Abstract base class for denoising methods.

    This class defines the interface that all denoising methods must implement.
    It provides a consistent API for different denoising algorithms and ensures
    compatibility with the YOLOv5 training and inference pipeline.
    """

    def __init__(self, **kwargs):
        """
        Initialize the denoising method with parameters.

        Args:
            **kwargs: Method-specific parameters
        """

        self.params = kwargs
        self._validate_params()

    @abstractmethod
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply denoising to the input image.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            **kwargs: Additional parameters for this specific call
        
        Returns:
            Denoised image as numpy array with same shape as input
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the denoising method.

        Returns:
            Method name as string
        """
        pass

    def _validate_params(self):
        """
        Validate the parameters passed to the constructor.
        Override this method in subclasses to add parameter validation.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current parameters of the denoising method.

        Returns:
            Dictionary of parameters
        """
        return self.params.copy()

    def set_params(self, **kwargs):
        """
        Update the parameters of the denoising method.

        Returns:
            String representation
        """
        self.params.update(kwargs)
        self._validate_params()

    def __repr__(self) -> str:
        """
        String representation of the denoising method.

        Returns:
            String representation
        """
        params_str = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"

class DenoisingRegistry:
    """
    Registry for denoising methods.

    This class maintains a registry of all available denoising methods and provides a factory 
    method to create instances of registered methods.
    """
    _methods: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, method_class: type):
        """
        Register a denoising method class.

        Args:
            name: Name to register the method under
            method_class: Class that inherits from DenoisingMethod
        """
        if not issubclass(method_class, DenoisingMethod):
            raise ValueError(f"Method class must inherit from DenoisingMethod, got {method_class}")
        cls._methods[name] = method_class

    @classmethod
    def create(cls, name: str, **kwargs) -> DenoisingMethod:
        """
        Create an instance of a registered denoising method.
        
        Args:
            name: Name of the registered method
            **kwargs: Parameters to pass to the method constructor
            
        Returns:
            Instance of the denoising method
            
        Raises:
            ValueError: If method name is not registered
        """
        if name not in cls._methods:
            available = ', '.join(cls._methods.keys())
            raise ValueError(f"Unknown denoising method '{name}'. Available methods: {available}")
        
        return cls._methods[name](**kwargs)

    @classmethod
    def list_methods(cls) -> list:
        """
        Get a list of all registered method names.
        
        Returns:
            List of registered method names
        """
        return list(cls._methods.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a method name is registered.
        
        Args:
            name: Method name to check
            
        Returns:
            True if method is registered, False otherwise
        """
        return name in cls._methods

def apply_denoising(image: np.ndarray, 
                   method: Union[str, DenoisingMethod] = 'fabf',
                   probability: float = 1.0,
                   **kwargs) -> np.ndarray:
    """
    Apply denoising to an image using the specified method.

    This is a convenience function that handles both string method names and DenoisingMethod instances.
    
    Args:
        image: Input image as numpy array
        method: Either a method name or a DenoisingMethod instance
        probability: Probability of applying the denoising (0.0 to 1.0)
        **kwargs: Additional parameters for the denoising method

    Returns:
        Denoised image (or original image if probability check fails)
    """
    import random
    
    # Check probability
    if probability < 1.0 and random.random() > probability:
        return image
    
    # Handle method specification
    if isinstance(method, str):
        denoiser = DenoisingRegistry.create(method, **kwargs)
    elif isinstance(method, DenoisingMethod):
        denoiser = method
    else:
        raise ValueError(f"Method must be string or DenoisingMethod instance, got {type(method)}")
    
    # Apply denoising
    return denoiser(image)

def ensure_float01(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is in the range [0, 1].

    Args:
        image: Input image as numpy array

    Returns:
        Image converted to float64 with values in [0, 1] range
    """
    image = image.astype(np.float64)
    if image.max() > 1.0:
        image /= 255.0
    return image


def validate_image(image: np.ndarray) -> None:
    """
    Validate that the image is in the correct format.
    
    Args:
        image: Input image to validate
        
    Raises:
        ValueError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(image)}")
    
    if image.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D (grayscale) or 3D (color), got {image.ndim}D")
    
    if image.ndim == 3 and image.shape[2] not in [1, 3]:
        raise ValueError(f"Color image must have 1 or 3 channels, got {image.shape[2]}")
    
    if image.size == 0:
        raise ValueError("Image cannot be empty")