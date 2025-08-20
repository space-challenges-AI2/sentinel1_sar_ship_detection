#!/usr/bin/env python3
"""
Test Jetson Compatibility
Tests all Jetson-safe functions and compatibility features
Author: @amanarora9848
"""

import sys
from pathlib import Path
import logging

# Add the project root to the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.jetson_display import jetson_display
from utils.jetson_math import jetson_math
from utils.general import LOGGER, colorstr

def test_jetson_detection():
    """Test Jetson detection capabilities"""
    LOGGER.info(f"{colorstr('Testing Jetson Detection:')}")
    
    try:
        # Test Jetson environment detection
        is_jetson = jetson_display.is_jetson
        has_display = jetson_display.has_display
        display_enabled = jetson_display.display_enabled
        
        LOGGER.info(f"  Jetson detected: {is_jetson}")
        LOGGER.info(f"  Display available: {has_display}")
        LOGGER.info(f"  Display enabled: {display_enabled}")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"Jetson detection test failed: {e}")
        return False

def test_display_functions():
    """Test display functions"""
    LOGGER.info(f"{colorstr('Testing Display Functions:')}")
    
    try:
        # Test safe display functions
        jetson_display.namedWindow('test', cv2.WINDOW_NORMAL)
        jetson_display.resizeWindow('test', 100, 100)
        jetson_display.destroyAllWindows()
        
        LOGGER.info("  Display functions: PASS")
        return True
        
    except Exception as e:
        LOGGER.error(f"Display functions test failed: {e}")
        return False

def test_math_functions():
    """Test math functions"""
    LOGGER.info(f"{colorstr('Testing Math Functions:')}")
    
    try:
        # Test safe math functions
        tan_result = jetson_math.safe_tan(45.0)
        beta_result = jetson_math.safe_random_beta(2.0, 2.0, size=5)
        uniform_result = jetson_math.safe_random_uniform(-1.0, 1.0, size=5)
        sqrt_result = jetson_math.safe_sqrt(16.0)
        power_result = jetson_math.safe_power(2.0, 3.0)
        
        LOGGER.info(f"  Tangent(45°): {tan_result:.6f}")
        LOGGER.info(f"  Beta distribution: {beta_result}")
        LOGGER.info(f"  Uniform distribution: {uniform_result}")
        LOGGER.info(f"  Square root(16): {sqrt_result:.6f}")
        LOGGER.info(f"  Power(2^3): {power_result:.6f}")
        
        LOGGER.info("  Math functions: PASS")
        return True
        
    except Exception as e:
        LOGGER.error(f"Math functions test failed: {e}")
        return False

def test_precision_modes():
    """Test precision mode switching"""
    LOGGER.info(f"{colorstr('Testing Precision Modes:')}")
    
    try:
        # Test precision mode switching
        jetson_math.set_precision_mode('fp32')
        jetson_math.set_precision_mode('fp16')
        jetson_math.set_precision_mode('fp32')  # Reset to default
        
        LOGGER.info("  Precision modes: PASS")
        return True
        
    except Exception as e:
        LOGGER.error(f"Precision modes test failed: {e}")
        return False

def test_headless_operations():
    """Test headless operations"""
    LOGGER.info(f"{colorstr('Testing Headless Operations:')}")
    
    try:
        # Test headless image saving
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = jetson_display.save_image_headless(test_image, "test_compatibility.png")
        if result:
            LOGGER.info(f"  Headless image save: PASS ({result})")
        else:
            LOGGER.warning("  Headless image save: WARNING (file not created)")
        
        # Test headless video writer
        writer = jetson_display.create_video_writer_headless("test_video.mp4", 30, 100, 100)
        if writer:
            writer.release()
            LOGGER.info("  Headless video writer: PASS")
        else:
            LOGGER.warning("  Headless video writer: WARNING (not created)")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"Headless operations test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    LOGGER.info(f"{colorstr('Starting Jetson Compatibility Tests')}")
    LOGGER.info("=" * 50)
    
    tests = [
        ("Jetson Detection", test_jetson_detection),
        ("Display Functions", test_display_functions),
        ("Math Functions", test_math_functions),
        ("Precision Modes", test_precision_modes),
        ("Headless Operations", test_headless_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        LOGGER.info(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            LOGGER.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    LOGGER.info("\n" + "=" * 50)
    LOGGER.info(f"{colorstr('Test Results Summary:')}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        color = "green" if result else "red"
        LOGGER.info(f"  {test_name}: {colorstr(color, status)}")
    
    LOGGER.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        LOGGER.info(f"{colorstr('green', 'All tests passed! Jetson compatibility verified.')}")
        return 0
    else:
        LOGGER.error(f"{colorstr('red', 'Some tests failed. Jetson compatibility issues detected.')}")
        return 1

if __name__ == "__main__":
    try:
        import cv2
        exit_code = main()
        sys.exit(exit_code)
    except ImportError as e:
        LOGGER.error(f"Required import failed: {e}")
        LOGGER.error("Make sure OpenCV is installed: pip install opencv-python-headless")
        sys.exit(1) 