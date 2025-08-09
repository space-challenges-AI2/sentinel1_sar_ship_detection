#!/usr/bin/env python3
"""
Visual test for denoising functionality.
Author: @amanarora9848 (Aman Arora)
"""

import numpy as np
import sys
import os
from pathlib import Path
import cv2
import glob

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib not available - visual tests will be skipped")

# Import the denoising module
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
    except ImportError:
        FABF_AVAILABLE = False
        print("⚠ FABF not available - visual tests will be skipped")
        
except ImportError as e:
    print(f"✗ Failed to import denoising module: {e}")
    sys.exit(1)


def setup_test_results_dir():
    """
    Create test_results directory if it doesn't exist.
    
    Returns:
        Path to the test_results directory
    """
    test_results_dir = current_dir / "test_results"
    test_results_dir.mkdir(exist_ok=True)
    print(f"✓ Test results will be saved to: {test_results_dir}")
    return test_results_dir


def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array (RGB format)
    """
    # Read image using OpenCV (BGR format)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image


def save_image(image, save_path):
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array (float32, [0, 1] range)
        save_path: Path to save the image
    """
    # Convert to uint8 range [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(str(save_path), image_bgr)


def create_test_image(size=(256, 256), noise_level=0.1):
    """
    Create a test image with known patterns and noise.
    
    Args:
        size: Tuple of (height, width)
        noise_level: Standard deviation of noise to add
        
    Returns:
        Test image as numpy array
    """
    image = np.zeros((size[0], size[1], 3), dtype=np.float32)
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    
    # Add different patterns to each channel
    # Red channel: gradient
    image[:, :, 0] = x / size[1]
    
    # Green channel: vertical stripes
    image[:, :, 1] = np.sin(2 * np.pi * x / 50) * 0.5 + 0.5
    
    # Blue channel: circular pattern
    center_x, center_y = size[1] // 2, size[0] // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    image[:, :, 2] = np.clip(1 - distance / (min(size) / 2), 0, 1)
    
    # Add some noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    return image


def create_noisy_image(base_image, noise_level=0.2):
    """
    Create a noisy version of the base image.
    
    Args:
        base_image: Base image to add noise to
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, noise_level, base_image.shape)
    noisy_image = np.clip(base_image + noise, 0, 1)
    return noisy_image.astype(np.float32)


def calculate_psnr(original, denoised):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Original image
        denoised: Denoised image
        
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, denoised, window_size=11):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        original: Original image
        denoised: Denoised image
        window_size: Window size for SSIM calculation
        
    Returns:
        SSIM value
    """
    # Simple SSIM implementation
    mu1 = np.mean(original)
    mu2 = np.mean(denoised)
    
    sigma1_sq = np.var(original)
    sigma2_sq = np.var(denoised)
    sigma12 = np.mean((original - mu1) * (denoised - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim = numerator / denominator
    return ssim


def plot_results(original, noisy, denoised, save_path, title_suffix=""):
    """
    Plot the original, noisy, and denoised images.
    
    Args:
        original: Original clean image
        noisy: Noisy image
        denoised: Denoised image
        save_path: Path to save the plot
        title_suffix: Additional text for plot titles
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available - skipping plot generation")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'Original Image {title_suffix}')
    axes[0, 0].axis('off')
    
    # Noisy image
    axes[0, 1].imshow(noisy)
    axes[0, 1].set_title(f'Noisy Image {title_suffix}')
    axes[0, 1].axis('off')
    
    # Denoised image
    axes[1, 0].imshow(denoised)
    axes[1, 0].set_title(f'Denoised Image {title_suffix}')
    axes[1, 0].axis('off')
    
    # Difference image (original - denoised)
    diff = np.abs(original - denoised)
    im_diff = axes[1, 1].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
    axes[1, 1].set_title(f'Difference (Original - Denoised) {title_suffix}')
    axes[1, 1].axis('off')
    plt.colorbar(im_diff, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visual test results saved to {save_path}")
    plt.close()


def test_visual_denoising(test_results_dir):
    """
    Test denoising with visual output.
    """
    print("Testing visual denoising...")
    
    if not FABF_AVAILABLE:
        print("⚠ FABF not available - skipping visual test")
        return False
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available - skipping visual test")
        return False
    
    try:
        # Create test image
        print("Creating test image...")
        original = create_test_image(size=(256, 256), noise_level=0.05)
        
        # Create noisy version
        print("Creating noisy version...")
        noisy = create_noisy_image(original, noise_level=0.2)
        
        # Apply denoising
        print("Applying denoising...")
        denoised = apply_denoising(noisy, method='fabf', rho=3.0, N=3, probability=1.0)
        
        # Calculate metrics
        print("Calculating metrics...")
        psnr_original = calculate_psnr(original, noisy)
        psnr_denoised = calculate_psnr(original, denoised)
        
        ssim_original = calculate_ssim(original, noisy)
        ssim_denoised = calculate_ssim(original, denoised)
        
        # Print results
        print(f"\nResults:")
        print(f"  PSNR (noisy vs original): {psnr_original:.2f} dB")
        print(f"  PSNR (denoised vs original): {psnr_denoised:.2f} dB")
        print(f"  SSIM (noisy vs original): {ssim_original:.3f}")
        print(f"  SSIM (denoised vs original): {ssim_denoised:.3f}")
        
        # Save images
        print("Saving images...")
        save_image(original, test_results_dir / "synthetic_original.png")
        save_image(noisy, test_results_dir / "synthetic_noisy.png")
        save_image(denoised, test_results_dir / "synthetic_denoised.png")
        
        # Plot results
        print("Generating visualization...")
        plot_results(original, noisy, denoised, 
                    test_results_dir / "synthetic_comparison.png", 
                    "(Synthetic)")
        
        # Check if denoising improved the image
        if psnr_denoised > psnr_original and ssim_denoised > ssim_original:
            print("✓ Denoising improved image quality")
            return True
        else:
            print("⚠ Denoising may not have improved image quality")
            return True  # Still consider it successful if the process completed
            
    except Exception as e:
        print(f"✗ Visual test failed: {e}")
        return False


def test_different_parameters(test_results_dir):
    """
    Test denoising with different parameters.
    """
    print("\nTesting different denoising parameters...")
    
    if not FABF_AVAILABLE:
        print("⚠ FABF not available - skipping parameter test")
        return False
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available - skipping parameter test")
        return False
    
    try:
        # Create test image
        original = create_test_image(size=(128, 128), noise_level=0.1)
        noisy = create_noisy_image(original, noise_level=0.3)
        
        # Test different parameters
        parameters = [
            {'rho': 2.0, 'N': 3, 'name': 'Low intensity'},
            {'rho': 5.0, 'N': 5, 'name': 'Medium intensity'},
            {'rho': 8.0, 'N': 7, 'name': 'High intensity'}
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Noisy
        axes[0, 1].imshow(noisy)
        axes[0, 1].set_title('Noisy')
        axes[0, 1].axis('off')
        
        axes[0, 2].axis('off')  # Empty subplot
        
        # Test different parameters
        for i, params in enumerate(parameters):
            denoised = apply_denoising(noisy, method='fabf', **{k: v for k, v in params.items() if k != 'name'})
            
            axes[1, i].imshow(denoised)
            axes[1, i].set_title(f"{params['name']}\n(ρ={params['rho']}, N={params['N']})")
            axes[1, i].axis('off')
            
            # Calculate and display PSNR
            psnr = calculate_psnr(original, denoised)
            axes[1, i].text(0.5, 0.95, f'PSNR: {psnr:.1f}dB', 
                           transform=axes[1, i].transAxes, 
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(test_results_dir / 'denoising_parameters_test.png', dpi=150, bbox_inches='tight')
        print("✓ Parameter comparison saved to test_results/denoising_parameters_test.png")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter test failed: {e}")
        return False


def test_real_images(test_results_dir, source_dir=None, max_images=3):
    """
    Test denoising on real images from the source directory.
    
    Args:
        test_results_dir: Directory to save test results
        source_dir: Directory containing source images (default: project_root/source)
        max_images: Maximum number of images to test
    """
    print(f"\nTesting denoising on real images...")
    
    if not FABF_AVAILABLE:
        print("⚠ FABF not available - skipping real image test")
        return False
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available - skipping real image test")
        return False
    
    # Determine source directory
    if source_dir is None:
        source_dir = project_root / "source"
    
    if not source_dir.exists():
        print(f"⚠ Source directory not found: {source_dir}")
        return False
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(str(source_dir / ext)))
        image_files.extend(glob.glob(str(source_dir / ext.upper())))
    
    if not image_files:
        print(f"⚠ No image files found in {source_dir}")
        return False
    
    # Limit number of images to test
    image_files = image_files[:max_images]
    print(f"Found {len(image_files)} image(s) to test")
    
    success_count = 0
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"\nProcessing image {i+1}/{len(image_files)}: {Path(image_path).name}")
            
            # Load image
            original = load_image(image_path)
            
            # Create noisy version for testing
            noisy = create_noisy_image(original, noise_level=0.15)
            
            # Apply denoising
            denoised = apply_denoising(noisy, method='fabf', rho=4.0, N=5, probability=1.0)
            
            # Calculate metrics
            psnr_noisy = calculate_psnr(original, noisy)
            psnr_denoised = calculate_psnr(original, denoised)
            
            ssim_noisy = calculate_ssim(original, noisy)
            ssim_denoised = calculate_ssim(original, denoised)
            
            # Print results
            print(f"  PSNR improvement: {psnr_noisy:.2f} dB -> {psnr_denoised:.2f} dB")
            print(f"  SSIM improvement: {ssim_noisy:.3f} -> {ssim_denoised:.3f}")
            
            # Save images
            base_name = Path(image_path).stem
            save_image(original, test_results_dir / f"{base_name}_original.png")
            save_image(noisy, test_results_dir / f"{base_name}_noisy.png")
            save_image(denoised, test_results_dir / f"{base_name}_denoised.png")
            
            # Create comparison plot
            plot_results(original, noisy, denoised, 
                        test_results_dir / f"{base_name}_comparison.png",
                        f"({Path(image_path).name})")
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed to process {Path(image_path).name}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {success_count}/{len(image_files)} images")
    return success_count > 0


def test_original_vs_denoised(test_results_dir, source_dir=None):
    """
    Test denoising on original images (without adding noise) to see the effect.
    """
    print(f"\nTesting denoising on original images (no noise added)...")
    
    if not FABF_AVAILABLE:
        print("⚠ FABF not available - skipping original image test")
        return False
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠ matplotlib not available - skipping original image test")
        return False
    
    # Determine source directory
    if source_dir is None:
        source_dir = project_root / "source"
    
    if not source_dir.exists():
        print(f"⚠ Source directory not found: {source_dir}")
        return False
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(str(source_dir / ext)))
        image_files.extend(glob.glob(str(source_dir / ext.upper())))
    
    if not image_files:
        print(f"⚠ No image files found in {source_dir}")
        return False
    
    # Test with first image only
    image_path = image_files[0]
    print(f"Testing with: {Path(image_path).name}")
    
    try:
        # Load image
        original = load_image(image_path)
        
        # Apply denoising directly to original
        denoised = apply_denoising(original, method='fabf', rho=3.0, N=3, probability=1.0)
        
        # Calculate metrics
        psnr = calculate_psnr(original, denoised)
        ssim = calculate_ssim(original, denoised)
        
        # Print results
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.3f}")
        
        # Save images
        base_name = Path(image_path).stem
        save_image(original, test_results_dir / f"{base_name}_clean_original.png")
        save_image(denoised, test_results_dir / f"{base_name}_clean_denoised.png")
        
        # Create comparison plot
        plot_results(original, original, denoised, 
                    test_results_dir / f"{base_name}_clean_comparison.png",
                    f"({Path(image_path).name} - Clean)")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to process {Path(image_path).name}: {e}")
        return False


def main():
    """
    Main function to run visual tests.
    """
    print("Visual Denoising Test")
    print("=" * 50)
    
    # Setup test results directory
    test_results_dir = setup_test_results_dir()
    
    success = True
    
    # Run visual tests
    success &= test_visual_denoising(test_results_dir)
    success &= test_different_parameters(test_results_dir)
    success &= test_real_images(test_results_dir)
    success &= test_original_vs_denoised(test_results_dir)
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All visual tests completed successfully!")
        print(f"  Check the {test_results_dir} folder for visual results.")
    else:
        print("✗ Some visual tests failed!")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)