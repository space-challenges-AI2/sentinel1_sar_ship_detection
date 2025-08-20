"""
Jetson-Compatible Math Functions
Handles ARM64 precision differences and provides safe alternatives
Author: @amanarora9848
"""

import math
import numpy as np
import torch
import logging
from typing import Union, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class JetsonMathHandler:
    """Handles math operations with Jetson compatibility"""
    
    def __init__(self):
        self.is_jetson = self._detect_jetson()
        self.precision_mode = 'fp32'  # Default precision
        
        if self.is_jetson:
            logger.info("Jetson detected - using ARM64-optimized math functions")
            self._setup_jetson_math()
        else:
            logger.info("x86_64 detected - using standard math functions")
    
    def _detect_jetson(self):
        """Detect if running on Jetson"""
        try:
            import platform
            if platform.machine() != 'aarch64':
                return False
            
            # Check for Jetson-specific files
            jetson_files = ['/etc/nv_tegra_release', '/sys/module/tegra_fuse/parameters/tegra_chip_id']
            return any(Path(f).exists() for f in jetson_files)
            
        except Exception:
            return False
    
    def _setup_jetson_math(self):
        """Setup Jetson-specific math optimizations"""
        try:
            # Set NumPy to use consistent precision
            np.set_printoptions(precision=6, suppress=True)
            
            # Set PyTorch to use deterministic algorithms where possible
            if hasattr(torch, 'set_deterministic'):
                torch.set_deterministic(True)
            
            # Use consistent random seed for reproducibility
            np.random.seed(42)
            if hasattr(torch, 'manual_seed'):
                torch.manual_seed(42)
                
        except Exception as e:
            logger.warning(f"Jetson math setup failed: {e}")
    
    def safe_tan(self, angle_degrees: float) -> float:
        """Safe tangent function with Jetson compatibility"""
        try:
            # Convert to radians
            angle_rad = math.radians(angle_degrees)
            
            # Use PyTorch for better precision on Jetson
            if self.is_jetson and hasattr(torch, 'tan'):
                return float(torch.tan(torch.tensor(angle_rad, dtype=torch.float32)))
            else:
                return math.tan(angle_rad)
                
        except Exception as e:
            logger.warning(f"Tangent calculation failed: {e}, using fallback")
            # Fallback to small angle approximation
            if abs(angle_degrees) < 10:
                return math.radians(angle_degrees)
            else:
                return 0.0
    
    def safe_random_beta(self, alpha: float, beta: float, size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Safe beta distribution with Jetson compatibility"""
        try:
            if self.is_jetson:
                # Use PyTorch for better precision on Jetson
                if hasattr(torch, 'distributions'):
                    from torch.distributions import Beta
                    dist = Beta(torch.tensor(alpha), torch.tensor(beta))
                    if size is None:
                        return dist.sample().numpy()
                    else:
                        return dist.sample(torch.Size(size)).numpy()
                else:
                    # Fallback to NumPy with error handling
                    result = np.random.beta(alpha, beta, size)
                    # Validate result
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        logger.warning("Beta distribution produced invalid values, using fallback")
                        return np.full(size if size else 1, 0.5)
                    return result
            else:
                return np.random.beta(alpha, beta, size)
                
        except Exception as e:
            logger.warning(f"Beta distribution failed: {e}, using uniform fallback")
            return np.random.uniform(0, 1, size)
    
    def safe_random_uniform(self, low: float, high: float, size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
        """Safe uniform distribution with Jetson compatibility"""
        try:
            result = np.random.uniform(low, high, size)
            
            # Validate result on Jetson
            if self.is_jetson:
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    logger.warning("Uniform distribution produced invalid values, using fallback")
                    return np.full(size if size else 1, (low + high) / 2)
            
            return result
            
        except Exception as e:
            logger.warning(f"Uniform distribution failed: {e}, using fallback")
            return np.full(size if size else 1, (low + high) / 2)
    
    def safe_interpolation(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        """Safe interpolation with Jetson compatibility"""
        try:
            # Ensure inputs are valid
            if len(xp) != len(fp):
                raise ValueError("xp and fp must have same length")
            
            if len(xp) == 0:
                return 0.0
            
            # Handle edge cases
            if x <= xp[0]:
                return float(fp[0])
            if x >= xp[-1]:
                return float(fp[-1])
            
            # Use NumPy interpolation
            result = np.interp(x, xp, fp)
            
            # Validate result on Jetson
            if self.is_jetson:
                if np.isnan(result) or np.isinf(result):
                    logger.warning("Interpolation produced invalid value, using nearest neighbor")
                    # Find nearest neighbor
                    idx = np.argmin(np.abs(xp - x))
                    return float(fp[idx])
            
            return float(result)
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, using fallback")
            return 0.0
    
    def safe_ceil(self, x: float) -> int:
        """Safe ceiling function with Jetson compatibility"""
        try:
            result = math.ceil(x)
            
            # Validate result on Jetson
            if self.is_jetson:
                if not isinstance(result, int) or result < 0:
                    logger.warning("Ceiling function produced invalid result, using fallback")
                    return max(0, int(x + 0.5))
            
            return result
            
        except Exception as e:
            logger.warning(f"Ceiling function failed: {e}, using fallback")
            return max(0, int(x + 0.5))
    
    def safe_sqrt(self, x: float) -> float:
        """Safe square root with Jetson compatibility"""
        try:
            if x < 0:
                logger.warning("Negative value passed to sqrt, using absolute value")
                x = abs(x)
            
            result = math.sqrt(x)
            
            # Validate result on Jetson
            if self.is_jetson:
                if np.isnan(result) or np.isinf(result):
                    logger.warning("Square root produced invalid result, using fallback")
                    return 0.0
            
            return result
            
        except Exception as e:
            logger.warning(f"Square root failed: {e}, using fallback")
            return 0.0
    
    def safe_power(self, base: float, exponent: float) -> float:
        """Safe power function with Jetson compatibility"""
        try:
            result = math.pow(base, exponent)
            
            # Validate result on Jetson
            if self.is_jetson:
                if np.isnan(result) or np.isinf(result):
                    logger.warning("Power function produced invalid result, using fallback")
                    if exponent == 2:
                        return base * base
                    elif exponent == 0.5:
                        return self.safe_sqrt(base)
                    else:
                        return 1.0
            
            return result
            
        except Exception as e:
            logger.warning(f"Power function failed: {e}, using fallback")
            return 1.0
    
    def set_precision_mode(self, mode: str):
        """Set precision mode for Jetson operations"""
        valid_modes = ['fp16', 'fp32', 'fp64']
        if mode not in valid_modes:
            logger.warning(f"Invalid precision mode {mode}, using fp32")
            mode = 'fp32'
        
        self.precision_mode = mode
        logger.info(f"Precision mode set to {mode}")
        
        if self.is_jetson:
            try:
                if mode == 'fp16':
                    # Set PyTorch to use FP16
                    if hasattr(torch, 'set_default_dtype'):
                        torch.set_default_dtype(torch.float16)
                elif mode == 'fp32':
                    # Set PyTorch to use FP32
                    if hasattr(torch, 'set_default_dtype'):
                        torch.set_default_dtype(torch.float32)
                        
            except Exception as e:
                logger.warning(f"Failed to set precision mode: {e}")

# Global instance
jetson_math = JetsonMathHandler()

# Convenience functions
def safe_tan(angle_degrees: float) -> float:
    """Safe tangent function"""
    return jetson_math.safe_tan(angle_degrees)

def safe_random_beta(alpha: float, beta: float, size=None) -> np.ndarray:
    """Safe beta distribution"""
    return jetson_math.safe_random_beta(alpha, beta, size)

def safe_random_uniform(low: float, high: float, size=None) -> np.ndarray:
    """Safe uniform distribution"""
    return jetson_math.safe_random_uniform(low, high, size)

def safe_interpolation(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Safe interpolation"""
    return jetson_math.safe_interpolation(x, xp, fp)

def safe_ceil(x: float) -> int:
    """Safe ceiling function"""
    return jetson_math.safe_ceil(x)

def safe_sqrt(x: float) -> float:
    """Safe square root"""
    return jetson_math.safe_sqrt(x)

def safe_power(base: float, exponent: float) -> float:
    """Safe power function"""
    return jetson_math.safe_power(base, exponent) 