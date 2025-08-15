"""
Pipeline Coordinator - Orchestrates the existing YOLOv5 pipeline
Author: @amanarora9848 (Aman Arora)
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os

# Import our pipeline services
from .ingest import SARIngestService
from .geo import GeoreferencingService
from .config import PipelineConfig
from .postproc import PostProcessingService
from .packager import PackagerService
from .health import HealthMonitor

class PipelineCoordinator:
    """
    Coordinates the pipeline using existing YOLOv5 tools.
    
    This service orchestrates the entire pipeline:
    1. Monitors for new SAR tiles
    2. Runs YOLO detection with denoising
    3. Converts detections to geographic coordinates
    4. Manages the workflow between stages
    """
    
    def __init__(self, 
                 config_file: str = "configs/flight.env",
                 **kwargs):
        """
        Initialize the pipeline coordinator.
        
        Args:
            config_file: Configuration file path
            **kwargs: Override parameters
        """
        # Load configuration
        self.config = PipelineConfig(config_file)
        
        # Get pipeline directory - use absolute path to /workspace in Docker
        # In Docker, the working directory should be /workspace
        if os.path.exists("/workspace"):
            # We're in Docker, use /workspace
            self.pipeline_dir = Path("/workspace")
        else:
            # We're running locally, use current working directory
            self.pipeline_dir = Path.cwd()
        
        # Use config values with fallbacks, ensuring paths are relative to pipeline directory
        self.ingest_dir = Path(kwargs.get('ingest_dir', 
                                        self.config.get('INGEST_DIR', 'test_ingest')))
        if not self.ingest_dir.is_absolute():
            self.ingest_dir = self.pipeline_dir / self.ingest_dir
            
        self.work_dir = Path(kwargs.get('work_dir', 
                                       self.config.get('WORK_DIR', 'test_work')))
        if not self.work_dir.is_absolute():
            self.work_dir = self.pipeline_dir / self.work_dir
            
        self.metadata_dir = Path(kwargs.get('metadata_dir', 
                                           self.config.get('METADATA_DIR', 'test_metadata')))
        if not self.metadata_dir.is_absolute():
            self.metadata_dir = self.pipeline_dir / self.metadata_dir
            
        self.detections_dir = Path(kwargs.get('detections_dir', 
                                             self.config.get('DETECTIONS_DIR', 'test_detections')))
        if not self.detections_dir.is_absolute():
            self.detections_dir = self.pipeline_dir / self.detections_dir
            
        self.thumbs_dir = Path(kwargs.get('thumbs_dir', 
                                         self.config.get('THUMBS_DIR', 'test_thumbs')))
        if not self.thumbs_dir.is_absolute():
            self.thumbs_dir = self.pipeline_dir / self.thumbs_dir
            
        self.outbox_dir = Path(kwargs.get('outbox_dir', 
                                         self.config.get('OUTBOX_DIR', 'test_outbox')))
        if not self.outbox_dir.is_absolute():
            self.outbox_dir = self.pipeline_dir / self.outbox_dir
            
        self.logs_dir = Path(kwargs.get('logs_dir', 
                                       self.config.get('LOGS_DIR', 'test_logs')))
        if not self.logs_dir.is_absolute():
            self.logs_dir = self.pipeline_dir / self.logs_dir
        
        # Create directories
        for dir_path in [self.ingest_dir, self.work_dir, self.metadata_dir, self.detections_dir, self.thumbs_dir, self.outbox_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate paths relative to the ingest_dir for other services
        # This ensures all test directories are in the same location
        base_dir = self.ingest_dir.parent
        georeferenced_dir = self.pipeline_dir / "test_georeferenced"
        postprocessed_dir = self.pipeline_dir / "test_postprocessed"
        
        # Initialize services with properly calculated test directories
        self.ingest_service = SARIngestService(
            ingest_dir=str(self.ingest_dir),
            work_dir=str(self.work_dir),
            metadata_dir=str(self.metadata_dir)
        )
        
        self.geo_service = GeoreferencingService(
            metadata_dir=str(self.metadata_dir),
            detections_dir=str(self.detections_dir),
            output_dir=str(georeferenced_dir)
        )
        
        self.postproc_service = PostProcessingService(
            detections_dir=str(self.detections_dir),
            output_dir=str(postprocessed_dir),
            thumbs_dir=str(self.thumbs_dir)
        )
        
        self.packager_service = PackagerService(
            input_dir=str(postprocessed_dir),
            thumbs_dir=str(self.thumbs_dir),
            outbox_dir=str(self.outbox_dir)
        )
        
        self.logger = self._setup_logging()
        
        # Pipeline state
        self.running = False
        self.stats = {
            'images_processed': 0,
            'detections_found': 0,
            'processing_time_total': 0.0,
            'last_processed': None,
            'thumbnails_generated': 0,  # Add missing key
            'packets_created': 0,        # Add missing key
            'total_detections': 0        # Add total detections counter
        }
        
        # Add health monitor
        self.health_monitor = HealthMonitor(
            pipeline_dir=self.pipeline_dir,
            logs_dir=self.logs_dir
        )
        
        # Load additional configuration values
        self.weights_path = Path(self.config.get('MODEL_WEIGHTS', 'weights/best.pt'))
        if not self.weights_path.is_absolute():
            # Since pipeline_dir is now current working directory, just use relative path
            self.weights_path = self.pipeline_dir / self.weights_path
        
        self.denoise_method = self.config.get('PREPROC_MODE', 'fabf')
        self.denoise_probability = float(self.config.get('FABF_RHO', '5.0'))
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the coordinator."""
        logger = logging.getLogger('PipelineCoordinator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start(self):
        """Start the pipeline coordinator."""
        try:
            self.logger.info("Starting SAR Pipeline Coordinator...")
            
            # Start health monitor first
            self.health_monitor.start_monitoring()
            
            # Start ingest service
            self.ingest_service.start()
            
            self.running = True
            self.logger.info("Pipeline coordinator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline coordinator: {e}")
            raise
    
    def stop(self):
        """Stop the pipeline coordinator."""
        try:
            self.logger.info("Stopping SAR Pipeline Coordinator...")
            
            # Stop health monitor
            self.health_monitor.stop_monitoring()
            
            # Stop ingest service
            self.ingest_service.stop()
            
            self.running = False
            self.logger.info("Pipeline coordinator stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping pipeline coordinator: {e}")
    
    def run_pipeline_cycle(self):
        """Run one cycle of the pipeline."""
        try:
            # Get next work item from ingest
            work_item = self.ingest_service.get_work_item()
            
            if work_item is None:
                return False  # No work to do
            
            self.logger.info(f"Processing: {work_item['file_path']}")
            start_time = time.time()
            
            # Step 1: Run YOLO detection with denoising
            detection_result = self._run_detection(work_item['file_path'])
            
            if detection_result['status'] == 'success':
                # Step 2: Georeference detections
                geo_result = self._georeference_detections(work_item['file_path'])
                
                # Handle case where no detections were found
                if geo_result is None:
                    # No detections found, but still mark as processed
                    processing_time = time.time() - start_time
                    self.stats['images_processed'] += 1
                    self.stats['processing_time_total'] += processing_time
                    self.stats['last_processed'] = time.time()
                    
                    self.logger.info(f"Pipeline completed for {work_item['file_path']} (no detections) in {processing_time:.2f}s")
                    
                    # Mark work item as complete
                    self.ingest_service.mark_work_complete(work_item)
                    return True
                
                if geo_result['status'] == 'success':
                    # Step 3: Post-process and generate thumbnails
                    postproc_result = self._post_process_detections()
                    
                    if postproc_result and postproc_result['status'] == 'success':
                        # Step 4: Create downlink packets
                        packet_result = self._create_downlink_packets()
                        
                        if packet_result:
                            # Update statistics
                            processing_time = time.time() - start_time
                            self.stats['images_processed'] += 1
                            self.stats['total_detections'] += geo_result.get('detections_count', 0)
                            self.stats['detections_found'] += geo_result.get('detections_count', 0)
                            self.stats['thumbnails_generated'] += postproc_result.get('thumbnails_count', 0)
                            self.stats['packets_created'] += packet_result.get('packets_created', 0)
                            self.stats['processing_time_total'] += processing_time
                            self.stats['last_processed'] = time.time()
                            
                            self.logger.info(f"Pipeline completed for {work_item['file_path']} in {processing_time:.2f}s")
                            self.logger.info(f"Statistics updated: {self.stats['images_processed']} images, {self.stats['total_detections']} detections")
                            
                            # Mark work item as complete
                            self.ingest_service.mark_work_complete(work_item)
                            
                            return True
                        else:
                            self.logger.error(f"Packet creation failed")
                    else:
                        self.logger.error(f"Post-processing failed")
                else:
                    self.logger.error(f"Georeferencing failed for {work_item['file_path']}")
            else:
                self.logger.error(f"Detection failed for {work_item['file_path']}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Pipeline cycle failed: {e}")
            return False
    
    def _run_detection(self, image_path: str) -> Dict[str, Any]:
        """Run YOLO detection on an image."""
        try:
            # Clean up the denoising method - remove any comments or extra text
            clean_denoise_method = self.denoise_method.split('#')[0].strip() if '#' in self.denoise_method else self.denoise_method
            
            # Auto-detect device - use CPU if CUDA not available
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = "0"  # Use first CUDA device
                self.logger.info("CUDA available, using GPU for detection")
            else:
                device = "cpu"  # Fallback to CPU
                self.logger.info("CUDA not available, using CPU for detection")
            
            # Build detect.py command with denoising
            cmd = [
                "python", "detect.py",
                "--weights", str(self.weights_path),
                "--source", image_path,
                "--project", str(self.detections_dir),
                "--name", "pipeline",
                "--exist-ok",
                "--device", device,  # ← THIS WAS MISSING!
                "--denoise", str(self.denoise_probability),
                "--denoise-method", clean_denoise_method,
                "--denoise-rho", "5.0",
                "--denoise-N", "5",
                "--denoise-sigma", "0.1",
                "--save-txt",
                "--save-conf"
            ]
            
            self.logger.debug(f"Running detection: {' '.join(cmd)}")
            self.logger.info(f"Using denoising method: '{clean_denoise_method}'")
            self.logger.info(f"Using device: {device}")
            self.logger.info(f"Denoising probability: {self.denoise_probability}")
            self.logger.info(f"Working directory: {self.pipeline_dir}")
            
            # Execute detect.py with fixed working directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.pipeline_dir
            )
            
            if result.returncode == 0:
                self.logger.info(f"Detection completed for {image_path}")
                # Verify that results were actually created
                detection_dir = self.detections_dir / "pipeline"
                if detection_dir.exists():
                    files_created = list(detection_dir.rglob("*"))
                    self.logger.info(f"Files created in detection directory: {len(files_created)}")
                    for file in files_created[:5]:  # Show first 5 files
                        self.logger.info(f"  - {file}")
                else:
                    self.logger.warning("Detection directory not created!")
                
                return {
                    'status': 'success',
                    'image_path': image_path,
                    'output': result.stdout
                }
            else:
                self.logger.error(f"Detection failed: {result.stderr}")
                return {
                    'status': 'failed',
                    'image_path': image_path,
                    'error': result.stderr
                }
                
        except Exception as e:
            self.logger.error(f"Error running detection: {e}")
            return {
                'status': 'error',
                'image_path': image_path,
                'error': str(e)
            }
    
    def _georeference_detections(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Georeference detection results."""
        try:
            # Find detection results file
            detection_file = self._find_detection_results(image_path)
            
            if detection_file:
                # Process detections through georeferencing service
                geo_result = self.geo_service.process_detections(detection_file)
                return geo_result
            else:
                self.logger.warning(f"No detection results found for {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in georeferencing: {e}")
            return None
    
    def _find_detection_results(self, image_path: str) -> Optional[str]:
        """Find the detection results file for an image."""
        try:
            image_path_obj = Path(image_path)
            base_name = image_path_obj.stem
            
            # Look for detection results in the output directory
            detection_dir = self.detections_dir / "pipeline"
            
            # Check for text results first
            txt_file = detection_dir / "labels" / f"{base_name}.txt"
            if txt_file.exists():
                return str(txt_file)
            
            # Check for JSON results
            json_file = detection_dir / f"{base_name}.json"
            if json_file.exists():
                return str(json_file)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding detection results: {e}")
            return None
    
    def _post_process_detections(self) -> Optional[Dict[str, Any]]:
        """Post-process detection results and generate thumbnails."""
        try:
            # Process detections through post-processing service
            postproc_result = self.postproc_service.process_detections(
                str(self.detections_dir),
                str(self.work_dir)  # Original images directory
            )
            return postproc_result
            
        except Exception as e:
            self.logger.error(f"Error in post-processing: {e}")
            return None
    
    def _create_downlink_packets(self) -> Optional[Dict[str, Any]]:
        """Create downlink packets from post-processed results."""
        try:
            # Create packets
            packets = self.packager_service.create_packets()
            
            if packets:
                self.logger.info(f"Created {len(packets)} downlink packets")
                return {
                    'status': 'success',
                    'packets_created': len(packets),
                    'total_size_mb': sum(p['total_size_bytes'] for p in packets) / (1024 * 1024)
                }
            else:
                self.logger.warning("No packets created")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating downlink packets: {e}")
            return None
    
    def run_continuous(self, cycle_interval: float = 5.0):
        """Run the pipeline continuously."""
        self.logger.info(f"Starting continuous pipeline with {cycle_interval}s cycle interval")
        
        try:
            while self.running:
                # Debug: Check work queue status
                ingest_status = self.ingest_service.get_status()
                queue_length = ingest_status.get('queue_length', 0)
                self.logger.info(f"Work queue length: {queue_length}")
                
                # Run one pipeline cycle
                cycle_result = self.run_pipeline_cycle()
                
                if not cycle_result:
                    self.logger.info("Pipeline cycle returned False - no work or processing failed")
                
                # Wait for next cycle
                time.sleep(cycle_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping pipeline")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            # Don't stop the pipeline on errors, just log them and continue
            time.sleep(cycle_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline coordinator."""
        status = super().get_status() if hasattr(super(), 'get_status') else {}
        
        return {
            'service': 'PipelineCoordinator',
            'status': 'running' if self.running else 'stopped',
            'ingest_directory': str(self.ingest_dir),
            'work_directory': str(self.work_dir),
            'output_directory': str(self.detections_dir),
            'weights_path': str(self.weights_path),
            'denoise_method': self.denoise_method,
            'denoise_probability': self.denoise_probability,
            'statistics': self.stats.copy(),
            'ingest_service': self.ingest_service.get_status(),
            'geo_service': self.geo_service.get_status(),
            'postproc_service': self.postproc_service.get_status(),
            'packager_service': self.packager_service.get_status(),
            'health_monitor': self.health_monitor.get_health_status()
        }