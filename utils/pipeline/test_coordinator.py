#!/usr/bin/env python3
"""
Test script for the Pipeline Coordinator
Tests the complete pipeline orchestration
"""

import os
import shutil
import time
from pathlib import Path
import sys

# Fix relative import issue by adding the project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up from utils/pipeline to project root
sys.path.insert(0, str(project_root))

from utils.pipeline.coordinator import PipelineCoordinator

def setup_test_environment():
    """Create test environment for pipeline testing."""
    print("Setting up pipeline test environment...")
    
    # Create test directories INSIDE utils/pipeline
    pipeline_dir = Path(__file__).parent
    
    test_dirs = [
        pipeline_dir / 'test_ingest',
        pipeline_dir / 'test_work', 
        pipeline_dir / 'test_detections',
        pipeline_dir / 'test_metadata',
        pipeline_dir / 'test_georeferenced',
        pipeline_dir / 'test_thumbs',
        pipeline_dir / 'test_outbox',
        pipeline_dir / 'test_logs'
    ]
    
    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)
        print(f"  Created: {test_dir.name}")
    
    # Copy test images from source directory
    source_dir = project_root / 'source'
    if source_dir.exists():
        test_images = ['test_image_good.png', '000019.jpg']
        for img in test_images:
            src = source_dir / img
            dst = pipeline_dir / 'test_ingest' / img
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied: {img}")
    
    print("Test environment ready!")
    return pipeline_dir

def test_pipeline_coordinator(pipeline_dir):
    """Test the pipeline coordinator."""
    print("\nTesting Pipeline Coordinator...")
    
    try:
        # Initialize coordinator with test directories
        coordinator = PipelineCoordinator(
            ingest_dir=pipeline_dir / 'test_ingest',
            work_dir=pipeline_dir / 'test_work',
            metadata_dir=pipeline_dir / 'test_metadata',
            detections_dir=pipeline_dir / 'test_detections',
            thumbs_dir=pipeline_dir / 'test_thumbs',
            outbox_dir=pipeline_dir / 'test_outbox',
            logs_dir=pipeline_dir / 'test_logs'
        )
        print("  Coordinator initialized")
        
        # Start coordinator
        coordinator.start()
        print("  Coordinator started")
        
        # Wait for initial processing
        print("  Waiting for initial processing...")
        time.sleep(3)
        
        # Check status
        status = coordinator.get_status()
        print(f"  Coordinator status: {status['status']}")
        print(f"  Ingest service: {status['ingest_service']}")
        print(f"  Geo service: {status['geo_service']}")
        
        # Run manual pipeline cycle
        print("  Running manual pipeline cycle...")
        success = coordinator.run_pipeline_cycle()
        
        if success:
            print("  Pipeline cycle completed successfully")
            print("  Final statistics:")
            # Get FRESH status after pipeline completion
            updated_status = coordinator.get_status()
            stats = updated_status['statistics']
            print(f"    - Images processed: {stats['images_processed']}")
            print(f"    - Detections found: {stats['detections_found']}")
        else:
            print("  Pipeline cycle failed")
        
        # Stop coordinator
        coordinator.stop()
        print("  Coordinator stopped")
        
        return success
        
    except Exception as e:
        print(f"  Error testing coordinator: {e}")
        return False

def cleanup_test_environment(pipeline_dir):
    """Clean up test directories."""
    print("\nCleaning up test environment...")
    
    # Don't remove any test directories - keep them all for inspection
    print("  Keeping all test directories for inspection:")
    test_dirs = [
        'test_ingest', 'test_work', 'test_detections', 'test_metadata',
        'test_georeferenced', 'test_thumbs', 'test_outbox', 'test_logs'
    ]
    
    for test_dir in test_dirs:
        full_path = pipeline_dir / test_dir
        if full_path.exists():
            print(f"    - {full_path}")
    
    print("  All test directories preserved for inspection")

def main():
    """Main test function."""
    print("Pipeline Coordinator Test Suite")
    print("=" * 50)
    
    # Setup test environment
    pipeline_dir = setup_test_environment()
    
    # Test coordinator
    success = test_pipeline_coordinator(pipeline_dir)
    
    # Report results
    print("\nTest Results:")
    if success:
        print("  Pipeline test: PASS")
    else:
        print("  Pipeline test: FAIL")
        print("\nPipeline test failed. Check the logs above.")
    
    # Cleanup (but keep directories)
    cleanup_test_environment(pipeline_dir)

if __name__ == "__main__":
    main()