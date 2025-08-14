#!/usr/bin/env python3
"""
SAR Ship Detection Pipeline - Demo
Author: @amanarora9848
"""

import time
import os
import threading
from pathlib import Path
from utils.pipeline import PipelineCoordinator

def run_demo():
    print("SAR Ship Detection Pipeline - Demo")
    print("=" * 60)
    print("Process:")
    print(" - AI-powered ship detection in satellite imagery")
    print(" - Real-time processing of SAR data")
    print(" - Automated georeferencing and packaging")
    print(" - Production-ready for satellite operations")
        
    # Add denoising control
    print("Denoising Options:")
    print("1. None (no denoising)")
    print("2. FABF (default)")
    
    choice = input("Select denoising method (1-2, default 2): ").strip()
    
    # Set denoising based on choice
    if choice == "1":
        denoise_method = "none"
        denoise_probability = 0.0
    else:  # default
        denoise_method = "fabf"
        denoise_probability = 1.0
    
    print(f"Using denoising: {denoise_method}")
    print()

    # Initialize pipeline with custom denoising
    print("Initializing pipeline...")
    coordinator = PipelineCoordinator()
    
    # Override denoising settings
    coordinator.denoise_method = denoise_method
    coordinator.denoise_probability = denoise_probability
    
    # Show configuration
    status = coordinator.get_status()
    print(f"Ingest directory: {status['ingest_directory']}")
    print(f"AI model: {status['weights_path']}")
    print(f"Denoising: {status['denoise_method']}")
    print()

    # Start pipeline
    print("Starting pipeline...")
    coordinator.start()

    # Start continuous processing in background thread
    print("Starting continuous pipeline processing...")
    pipeline_thread = threading.Thread(
        target=coordinator.run_continuous, 
        args=(2.0,),  # 2 second cycle interval
        daemon=True
    )
    pipeline_thread.start()

    print("Pipeline running! Now demonstrating capabilities...")
    print()

    # Process all images from the source/ directory
    import shutil
    source_dir = Path("source")
    ingest_dir = Path(coordinator.get_status()['ingest_directory'])
    if source_dir.exists() and source_dir.is_dir():
        image_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]
        if image_files:
            print(f"Found {len(image_files)} image(s) in {source_dir}. Copying to ingest directory...")
            for img_path in image_files:
                shutil.copy2(img_path, ingest_dir)
                print(f"Copied {img_path.name} to ingest directory")
            print("Waiting for pipeline to process the images...")
            time.sleep(3)  # Give pipeline time to detect and process the files
        else:
            print(f"No images found in {source_dir}. Pipeline will wait for images to be added to ingest directory.")
    else:
        print("No source directory found. Pipeline will wait for images to be added to ingest directory.")

    # Demo loop - now the pipeline should have work to do
    for i in range(5):
        status = coordinator.get_status()
        stats = status['statistics']
        
        print(f"Status Update {i+1}:")
        print(f" - Images processed: {stats['images_processed']}")
        print(f" - Ships detected: {stats['total_detections']}")
        print(f" - Processing time: {stats['processing_time_total']:.1f}s")
        print(f" - System status: {status['status']}")
        print()
        
        time.sleep(2)

    # Wait for pipeline to process all images
    print("Waiting for pipeline to process all images...")
    max_wait_time = 120  # Maximum wait time in seconds (2 minutes)
    start_wait = time.time()

    while time.time() - start_wait < max_wait_time:
        status = coordinator.get_status()
        stats = status['statistics']
        ingest_status = status['ingest_service']
        
        print(f"Current status: {stats['images_processed']} images processed, {ingest_status['queue_length']} in queue")
        
        # Check if all images have been processed (8 total)
        if stats['images_processed'] >= 8:
            print("All images processed!")
            break
        
        time.sleep(3)
    else:
        print("Timeout waiting for all images to be processed")

    # Get final stats
    final_status = coordinator.get_status()
    final_stats = final_status['statistics']
    print(f"Final Results:")
    print(f" - Total images: {final_stats['images_processed']}")
    print(f" - Total ships: {final_stats['total_detections']}")
    print(f" - Total processing time: {final_stats['processing_time_total']:.1f}s")
    print()

    print("Stopping pipeline...")
    coordinator.stop()
    
    print("Demo complete!")
    print()
    # print("Next steps:")
    # print(" - Deploy to satellite or ground station")
    # print(" - Connect to real SAR data feeds")
    # print(" - Monitor ship detection in real-time")
    # print(" - Generate automated alerts for maritime security")

if __name__ == "__main__":
    run_demo()