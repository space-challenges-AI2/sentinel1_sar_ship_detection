#!/usr/bin/env python3
"""
Simple test script for denoising functionality.
Author: @amanarora9848 (Aman Arora)
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Now import the denoising module
try:
    from utils.denoising.base import (
        DenoisingMethod,
        DenoisingRegistry,
        apply_denoising,
        ensure_float01,
        validate_image
    )
    
    # Try to import FABF if available
    try:
        from utils.denoising.fabf import adaptive_bilateral_filter
        FABF_AVAILABLE = True
        print("✓ FABF available")
    except ImportError as e:
        FABF_AVAILABLE = False
        print(f"⚠ FABF not available - some tests will be skipped: {e}")
        
except ImportError as e:
    print(f"✗ Failed to import denoising module: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic denoising functionality"""
    print("Testing basic denoising functionality...")
    
    # Create test image
    test_image = np.random.rand(64, 64, 3).astype(np.float32)
    print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
    
    # Test validation
    try:
        validate_image(test_image)
        print("✓ Image validation passed")
    except Exception as e:
        print(f"✗ Image validation failed: {e}")
        return False
    
    # Test float conversion
    try:
        float_image = ensure_float01(test_image)
        print(f"✓ Float conversion passed, max value: {float_image.max()}")
    except Exception as e:
        print(f"✗ Float conversion failed: {e}")
        return False
    
    # Test denoising if FABF is available
    if FABF_AVAILABLE:
        try:
            result = apply_denoising(test_image, method='fabf', rho=3.0, probability=1.0)
            print(f"✓ Denoising passed, result shape: {result.shape}")
            
            # Check if denoising actually changed the image
            if not np.allclose(result, test_image, rtol=1e-5):
                print("✓ Denoising produced different result (as expected)")
            else:
                print("⚠ Denoising produced identical result (may be expected for some cases)")
                
        except Exception as e:
            print(f"✗ Denoising failed: {e}")
            return False
    else:
        print("⚠ Skipping denoising test - FABF not available")
    
    return True

def test_registry():
    """Test the registry functionality"""
    print("\nTesting registry functionality...")
    
    try:
        # List available methods
        methods = DenoisingRegistry.list_methods()
        print(f"✓ Available methods: {methods}")
        
        if 'fabf' in methods:
            # Test creating a denoiser
            denoiser = DenoisingRegistry.create('fabf', rho=2.0)
            print(f"✓ Created denoiser: {denoiser}")
            
            # Test the denoiser
            test_image = np.random.rand(32, 32).astype(np.float32)
            result = denoiser(test_image)
            print(f"✓ Denoiser test passed, result shape: {result.shape}")
        else:
            print("⚠ FABF method not available")
            
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        return False
    
    return True

def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    # Test invalid image
    try:
        validate_image("invalid")
        print("✗ Should have raised ValueError for invalid image")
        return False
    except ValueError:
        print("✓ Invalid image correctly rejected")
    
    # Test invalid method
    try:
        apply_denoising(np.random.rand(32, 32), method='nonexistent')
        print("✗ Should have raised ValueError for invalid method")
        return False
    except ValueError:
        print("✓ Invalid method correctly rejected")
    
    return True

def main():
    """Main test function"""
    print("Testing Denoising Wrapper")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_basic_functionality()
    success &= test_registry()
    success &= test_error_handling()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)