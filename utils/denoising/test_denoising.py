# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Test suite for denoising utilities
Author: @amanarora9848 (Aman Arora)
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import the denoising module
from .base import (
    DenoisingMethod,
    DenoisingRegistry,
    apply_denoising,
    ensure_float01,
    validate_image
)

# Import FABF if available
try:
    from .fabf import adaptive_bilateral_filter
    from . import FABFDenoiser
    FABF_AVAILABLE = True
except ImportError:
    FABF_AVAILABLE = False


class TestDenoisingMethod(unittest.TestCase):
    """Test the abstract base class DenoisingMethod"""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that DenoisingMethod cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            DenoisingMethod()


class TestFABFDenoiser(unittest.TestCase):
    """Test the FABF denoising method"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        if not FABF_AVAILABLE:
            raise unittest.SkipTest("FABF not available")
        
        # Create test images
        cls.grayscale_image = np.random.rand(64, 64).astype(np.float32)
        cls.color_image = np.random.rand(64, 64, 3).astype(np.float32)
        cls.uint8_image = (np.random.rand(64, 64) * 255).astype(np.uint8)
        
    def test_fabf_denoiser_initialization(self):
        """Test FABF denoiser initialization"""
        denoiser = FABFDenoiser(rho=3.0, N=3)
        self.assertEqual(denoiser.get_name(), 'fabf')
        self.assertEqual(denoiser.get_params()['rho'], 3.0)
        self.assertEqual(denoiser.get_params()['N'], 3)
    
    def test_fabf_denoiser_call_grayscale(self):
        """Test FABF denoiser on grayscale image"""
        denoiser = FABFDenoiser(rho=2.0, N=3)
        result = denoiser(self.grayscale_image)
        
        self.assertEqual(result.shape, self.grayscale_image.shape)
        self.assertEqual(result.dtype, self.grayscale_image.dtype)
        self.assertTrue(np.allclose(result, result, rtol=1e-5))  # Check for NaN/Inf
    
    def test_fabf_denoiser_call_color(self):
        """Test FABF denoiser on color image"""
        denoiser = FABFDenoiser(rho=2.0, N=3)
        result = denoiser(self.color_image)
        
        self.assertEqual(result.shape, self.color_image.shape)
        self.assertEqual(result.dtype, self.color_image.dtype)
        self.assertTrue(np.allclose(result, result, rtol=1e-5))
    
    def test_fabf_denoiser_call_uint8(self):
        """Test FABF denoiser on uint8 image"""
        denoiser = FABFDenoiser(rho=2.0, N=3)
        result = denoiser(self.uint8_image)
        
        self.assertEqual(result.shape, self.uint8_image.shape)
        self.assertEqual(result.dtype, self.uint8_image.dtype)
        self.assertTrue(np.allclose(result, result, rtol=1e-5))
    
    def test_fabf_denoiser_parameter_override(self):
        """Test FABF denoiser parameter override in call"""
        denoiser = FABFDenoiser(rho=5.0, N=5)
        result1 = denoiser(self.grayscale_image)
        result2 = denoiser(self.grayscale_image, rho=2.0, N=3)
        
        # Results should be different due to different parameters
        self.assertFalse(np.allclose(result1, result2))


class TestDenoisingRegistry(unittest.TestCase):
    """Test the DenoisingRegistry class"""
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        registry = DenoisingRegistry()
        self.assertEqual(registry.list_methods(), [])
    
    def test_register_method(self):
        """Test method registration"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        # Test registration
        DenoisingRegistry.register('fabf', FABFDenoiser)
        self.assertIn('fabf', DenoisingRegistry.list_methods())
    
    def test_create_method(self):
        """Test method creation"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        DenoisingRegistry.register('fabf', FABFDenoiser)
        denoiser = DenoisingRegistry.create('fabf', rho=3.0)
        self.assertIsInstance(denoiser, FABFDenoiser)
        self.assertEqual(denoiser.get_params()['rho'], 3.0)
    
    def test_create_nonexistent_method(self):
        """Test creating non-existent method"""
        with self.assertRaises(ValueError):
            DenoisingRegistry.create('nonexistent_method')
    
    def test_register_invalid_method(self):
        """Test registering invalid method class"""
        class InvalidClass:
            pass
        
        with self.assertRaises(ValueError):
            DenoisingRegistry.register('invalid', InvalidClass)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_ensure_float01(self):
        """Test ensure_float01 function"""
        # Test float32 image in [0, 1] range
        img_float = np.random.rand(32, 32).astype(np.float32)
        result = ensure_float01(img_float)
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
        
        # Test uint8 image
        img_uint8 = (np.random.rand(32, 32) * 255).astype(np.uint8)
        result = ensure_float01(img_uint8)
        self.assertEqual(result.dtype, np.float64)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
    
    def test_validate_image(self):
        """Test validate_image function"""
        # Test valid grayscale image
        valid_grayscale = np.random.rand(32, 32)
        validate_image(valid_grayscale)  # Should not raise
        
        # Test valid color image
        valid_color = np.random.rand(32, 32, 3)
        validate_image(valid_color)  # Should not raise
        
        # Test invalid image type
        with self.assertRaises(ValueError):
            validate_image("not an image")
        
        # Test invalid dimensions
        with self.assertRaises(ValueError):
            validate_image(np.random.rand(32, 32, 32, 32))
        
        # Test invalid channels
        with self.assertRaises(ValueError):
            validate_image(np.random.rand(32, 32, 5))
        
        # Test empty image
        with self.assertRaises(ValueError):
            validate_image(np.array([]))


class TestApplyDenoising(unittest.TestCase):
    """Test the apply_denoising function"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_image = np.random.rand(32, 32).astype(np.float32)
    
    def test_apply_denoising_with_string_method(self):
        """Test apply_denoising with string method name"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        result = apply_denoising(self.test_image, method='fabf', rho=2.0)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_apply_denoising_with_denoiser_instance(self):
        """Test apply_denoising with denoiser instance"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        denoiser = FABFDenoiser(rho=2.0)
        result = apply_denoising(self.test_image, method=denoiser)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_apply_denoising_probability(self):
        """Test apply_denoising with probability"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        # Test with probability 0 (should return original)
        result = apply_denoising(self.test_image, method='fabf', probability=0.0)
        np.testing.assert_array_equal(result, self.test_image)
        
        # Test with probability 1 (should apply denoising)
        result = apply_denoising(self.test_image, method='fabf', probability=1.0)
        self.assertEqual(result.shape, self.test_image.shape)
    
    def test_apply_denoising_invalid_method(self):
        """Test apply_denoising with invalid method"""
        with self.assertRaises(ValueError):
            apply_denoising(self.test_image, method=123)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_image = np.random.rand(64, 64, 3).astype(np.float32)
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end denoising pipeline"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        # Test the complete pipeline
        denoiser = FABFDenoiser(rho=3.0, N=3)
        result = denoiser(self.test_image)
        
        # Basic checks
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, self.test_image.dtype)
        self.assertTrue(np.allclose(result, result, rtol=1e-5))
        
        # Check that result is different from input (denoising occurred)
        self.assertFalse(np.allclose(result, self.test_image, rtol=1e-5))
    
    def test_registry_integration(self):
        """Test registry integration"""
        if not FABF_AVAILABLE:
            self.skipTest("FABF not available")
        
        # Test that FABF is automatically registered
        self.assertIn('fabf', DenoisingRegistry.list_methods())
        
        # Test creating through registry
        denoiser = DenoisingRegistry.create('fabf', rho=2.0)
        self.assertIsInstance(denoiser, FABFDenoiser)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDenoisingMethod))
    test_suite.addTest(unittest.makeSuite(TestDenoisingRegistry))
    test_suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    test_suite.addTest(unittest.makeSuite(TestApplyDenoising))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    if FABF_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(TestFABFDenoiser))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)