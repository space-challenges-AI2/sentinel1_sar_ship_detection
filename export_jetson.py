#!/usr/bin/env python3
"""
Export YOLO model to Jetson-optimized formats
Optimizes for ARM64 architecture and TensorRT acceleration
Author: @amanarora9848
"""

import argparse
import torch
import os
from pathlib import Path
from utils.general import LOGGER, colorstr

def check_jetson_environment():
    """Check if running in Jetson environment"""
    try:
        import platform
        is_arm64 = platform.machine() == 'aarch64'
        
        # Check for Jetson-specific files
        jetson_files = ['/etc/nv_tegra_release', '/sys/module/tegra_fuse/parameters/tegra_chip_id']
        is_jetson = any(Path(f).exists() for f in jetson_files)
        
        if is_arm64 and is_jetson:
            LOGGER.info(f"{colorstr('Jetson detected:')} ARM64 architecture with Jetson hardware")
            return True
        elif is_arm64:
            LOGGER.info(f"{colorstr('ARM64 detected:')} Running on ARM64 architecture")
            return True
        else:
            LOGGER.info(f"{colorstr('x86_64 detected:')} Running on x86_64 architecture")
            return False
            
    except Exception as e:
        LOGGER.warning(f"Could not detect architecture: {e}")
        return False

def export_for_jetson(weights_path, output_dir="jetson_export", include_formats=None):
    """Export model for Jetson deployment"""
    
    if include_formats is None:
        include_formats = ['onnx', 'engine']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    LOGGER.info(f"Exporting {weights_path} for Jetson...")
    
    exported_files = []
    
    # 1. Export to ONNX (required for TensorRT)
    if 'onnx' in include_formats:
        onnx_path = export_to_onnx(weights_path, output_path)
        if onnx_path:
            exported_files.append(('ONNX', onnx_path))
    
    # 2. Convert to TensorRT (if available)
    if 'engine' in include_formats:
        trt_path = export_to_tensorrt(weights_path, output_path)
        if trt_path:
            exported_files.append(('TensorRT', trt_path))
    
    # 3. Create Jetson-optimized PyTorch version
    jetson_pt = create_jetson_optimized_pt(weights_path, output_path)
    if jetson_pt:
        exported_files.append(('PyTorch (Jetson)', jetson_pt))
    
    LOGGER.info(f"\n{colorstr('Export complete!')} Files saved to {output_path}")
    for format_name, file_path in exported_files:
        LOGGER.info(f"{format_name}: {file_path}")
    
    return exported_files

def export_to_onnx(weights_path, output_dir):
    """Export to ONNX format"""
    try:
        LOGGER.info(f"{colorstr('ONNX:')} starting export...")
        
        # Use your existing export.py functionality
        from export import export_onnx
        
        # Load model and dummy input with weights_only=False for compatibility
        model = torch.load(weights_path, map_location='cpu', weights_only=False)
        if 'model' in model:
            model = model['model']
        
        # Ensure model is in FP32 for ONNX export
        model = model.float()
        model.eval()
        
        # Create dummy input for ONNX export
        dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
        
        # Export to ONNX
        onnx_path = output_dir / "model.onnx"
        export_onnx(model, dummy_input, onnx_path, opset=12, dynamic=True, simplify=True)
        
        # Verify file was created
        if onnx_path.exists() and onnx_path.stat().st_size > 0:
            LOGGER.info(f"{colorstr('ONNX:')} export completed: {onnx_path}")
            return onnx_path
        else:
            LOGGER.error("ONNX file was not created or is empty")
            return None
        
    except Exception as e:
        LOGGER.error(f"ONNX export failed: {e}")
        return None

def export_to_tensorrt(weights_path, output_dir):
    """Export to TensorRT format"""
    try:
        LOGGER.info(f"{colorstr('TensorRT:')} starting export...")
        
        # Check if TensorRT is available
        try:
            import tensorrt as trt
            LOGGER.info(f"TensorRT {trt.__version__} available")
        except ImportError:
            LOGGER.warning("TensorRT not available, skipping TensorRT export")
            return None
        
        # First export to ONNX (required for TensorRT)
        onnx_path = output_dir / "model.onnx"
        if not onnx_path.exists():
            LOGGER.info("ONNX file not found, creating it first...")
            onnx_result = export_to_onnx(weights_path, output_dir)
            if not onnx_result:
                LOGGER.error("Failed to create ONNX file for TensorRT export")
                return None
        
        # Use your existing export.py functionality
        from export import export_engine
        
        # Export to TensorRT with proper parameters
        trt_path = output_dir / "model.engine"
        export_engine(None, onnx_path, half=True, dynamic=True, workspace=2)
        
        # Check if file was created
        if trt_path.exists() and trt_path.stat().st_size > 0:
            LOGGER.info(f"{colorstr('TensorRT:')} export completed: {trt_path}")
            return trt_path
        else:
            LOGGER.error("TensorRT file was not created or is empty")
            return None
        
    except Exception as e:
        LOGGER.error(f"TensorRT export failed: {e}")
        return None

def create_jetson_optimized_pt(weights_path, output_dir):
    """Create Jetson-optimized PyTorch model"""
    try:
        LOGGER.info(f"{colorstr('PyTorch:')} creating Jetson-optimized version...")
        
        # Load the model with weights_only=False for compatibility
        model_data = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Apply Jetson-specific optimizations
        if 'model' in model_data:
            model = model_data['model']
            
            # Set to evaluation mode
            model.eval()
            
            # Apply optimizations for Jetson
            model = optimize_model_for_jetson(model)
            
            # Update model data
            model_data['model'] = model
        
        # Save optimized model
        jetson_path = output_dir / "model_jetson_optimized.pt"
        torch.save(model_data, jetson_path)
        
        LOGGER.info(f"{colorstr('PyTorch:')} Jetson-optimized model saved: {jetson_path}")
        return jetson_path
        
    except Exception as e:
        LOGGER.error(f"PyTorch optimization failed: {e}")
        return None

def optimize_model_for_jetson(model):
    """Apply Jetson-specific optimizations to PyTorch model"""
    try:
        # Set optimal settings for Jetson
        model.half()  # Use FP16 for better Jetson performance
        
        # Fuse batch norm layers for faster inference
        if hasattr(torch, 'jit'):
            model = torch.jit.script(model)
        
        # Set to evaluation mode
        model.eval()
        
        LOGGER.info("Applied Jetson optimizations: FP16, JIT compilation")
        return model
        
    except Exception as e:
        LOGGER.warning(f"Some Jetson optimizations failed: {e}")
        return model

def main():
    parser = argparse.ArgumentParser(description='Export YOLO model for Jetson deployment')
    parser.add_argument('--weights', type=str, default='runs/train/experiment4/weights/best.pt', 
                       help='Path to weights file')
    parser.add_argument('--output', type=str, default='jetson_export', 
                       help='Output directory')
    parser.add_argument('--include', nargs='+', default=['onnx', 'engine'], 
                       choices=['onnx', 'engine', 'pt'], 
                       help='Formats to export')
    
    args = parser.parse_args()
    
    # Check Jetson environment
    is_jetson = check_jetson_environment()
    
    if not is_jetson:
        LOGGER.warning("Not running on Jetson - some optimizations may not be available")
    
    # Export model
    exported_files = export_for_jetson(args.weights, args.output, args.include)
    
    if exported_files:
        LOGGER.info(f"\n{colorstr('Success:')} Exported {len(exported_files)} format(s)")
        LOGGER.info("Next steps:")
        LOGGER.info("1. Copy exported files to Jetson")
        LOGGER.info("2. Build Jetson Docker image: ./utils/docker/build-jetson.sh")
        LOGGER.info("3. Run container with GPU access")
    else:
        LOGGER.error("Export failed - no files were created")

if __name__ == "__main__":
    main() 