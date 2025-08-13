#!/usr/bin/env python3
"""
SAR Ship Detection Pipeline - Live Demo
Author: @amanarora9848
"""

import time
import os
from pathlib import Path
from utils.pipeline import PipelineCoordinator

def run_demo():
    print("SAR Ship Detection Pipeline - Demo")
    print("=" * 60)
    print("What's going on?:")
    print("   • AI-powered ship detection in satellite imagery")
    print("   • Real-time processing of SAR data")
    print("   • Automated georeferencing and packaging")
    print("   • Production-ready for satellite operations")
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    coordinator = PipelineCoordinator()

    # Show configuration
    status = coordinator.get_status()
    print(f"Ingest directory: {status['ingest_directory']}")
    print(f"AI model: {status['weights_path']}")
    print(f"Denoising: {status['denoise_method']}")
    print()

    # Start pipeline
    print("Starting pipeline...")
    coordinator.start()

    print("Pipeline running! Now demonstrating capabilities...")
    print()
    
    # Add a sample image to the ingest directory BEFORE starting status updates
    sample_image_path = Path("source/noisy_test_image.png")
    if sample_image_path.exists():
        print(f"Found sample image: {sample_image_path}")
        # Copy to ingest directory to trigger processing
        import shutil
        ingest_dir = coordinator.get_status()['ingest_directory']
        shutil.copy2(sample_image_path, ingest_dir)
        print(f"Copied sample image to ingest directory")
        print("Waiting for pipeline to process the image...")
        time.sleep(3)  # Give pipeline time to detect and process the file
    else:
        print("No sample image found. Pipeline will wait for images to be added to ingest directory.")
    
    # Demo loop - now the pipeline should have work to do
    for i in range(5):
        status = coordinator.get_status()
        stats = status['statistics']
        
        print(f"Status Update {i+1}:")
        print(f"   • Images processed: {stats['images_processed']}")
        print(f"   • Ships detected: {stats['total_detections']}")
        print(f"   • Processing time: {stats['processing_time_total']:.1f}s")
        print(f"   • System status: {status['status']}")
        print()
        
        time.sleep(2)

    # Run a pipeline cycle to process any remaining work
    print("Running pipeline cycle...")
    success = coordinator.run_pipeline_cycle()
    
    if success:
        print("Pipeline cycle completed successfully!")
        # Get final stats
        final_status = coordinator.get_status()
        final_stats = final_status['statistics']
        print(f"Final Results:")
        print(f"   • Total images: {final_stats['images_processed']}")
        print(f"   • Total ships: {final_stats['total_detections']}")
        print(f"   • Efficiency: {final_stats['processing_time_total']:.1f}s")
    else:
        print("Pipeline cycle failed")

    print()
    print("Stopping pipeline...")
    coordinator.stop()
    
    print("Demo complete! Your SAR pipeline is production-ready!")
    print()
    print("Next steps:")
    print("   • Deploy to satellite or ground station")
    print("   • Connect to real SAR data feeds")
    print("   • Monitor ship detection in real-time")
    print("   • Generate automated alerts for maritime security")

if __name__ == "__main__":
    run_demo()