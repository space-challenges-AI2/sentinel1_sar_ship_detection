"""
SAR Ingest Service - Watches for incoming SAR tiles and creates metadata sidecars
Author: @amanarora9848 (Aman Arora)
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import numpy as np
from PIL import Image
import yaml

class SARIngestService:
    """
    Monitors /ingest directory for new SAR tiles and creates metadata sidecars.
    
    This service:
    1. Watches for new files in the ingest directory
    2. Validates SAR tile format and integrity
    3. Creates metadata sidecars (.meta.json) with geospatial info
    4. Moves validated tiles to /work/ready for processing
    5. Maintains a work queue for the preprocessing stage
    """

    def __init__(self, 
                 ingest_dir: str = "/ingest",
                 work_dir: str = "/work/ready",
                 metadata_dir: str = "/work/metadata",
                 supported_formats: List[str] = None,
                 max_file_size_mb: int = 100,
                 health_check_interval: int = 30):
        """
        Initialize the SAR ingest service.
        
        Args:
            ingest_dir: Directory to watch for incoming SAR tiles
            work_dir: Directory to move validated tiles for processing
            metadata_dir: Directory to store metadata sidecars
            supported_formats: List of supported image formats
            max_file_size_mb: Maximum file size in MB
            health_check_interval: Health check interval in seconds
        """
        self.ingest_dir = Path(ingest_dir)
        self.work_dir = Path(work_dir)
        self.metadata_dir = Path(metadata_dir)
        self.max_file_size_mb = max_file_size_mb
        self.health_check_interval = health_check_interval
        # Default supported formats for SAR imagery
        self.supported_formats = supported_formats or [
            'jpg', 'jpeg', 'png', 'tif', 'tiff', 'geotiff'
        ]
        
        # Create directories if they don't exist
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file watcher
        self.observer = Observer()
        self.event_handler = SARFileHandler(self)
        self.observer.schedule(self.event_handler, str(self.ingest_dir), recursive=False)
        
        # Work queue for preprocessing stage
        self.work_queue = []
        self.queue_lock = threading.Lock()

        # Statistics and health
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'last_processed': None,
            'queue_length': 0
        }

        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the ingest service."""
        logger = logging.getLogger('SARIngest')
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
        """Start the file watcher."""
        try:
            self.observer.start()
            self.logger.info(f"Started watching {self.ingest_dir} for SAR tiles")
            
            # Process any existing files
            self._process_existing_files()
            
        except Exception as e:
            self.logger.error(f"Failed to start ingest service: {e}")
            raise
    
    def stop(self):
        """Stop the file watcher."""
        try:
            self.observer.stop()
            self.observer.join()
            self.logger.info("Stopped SAR ingest service")
        except Exception as e:
            self.logger.error(f"Error stopping ingest service: {e}")
    
    def _process_existing_files(self):
        """Process any files that already exist in the ingest directory."""
        for file_path in self.ingest_dir.glob("*"):
            if file_path.is_file():
                self.logger.info(f"Processing existing file: {file_path}")
                self._process_file(file_path)
    
    def _process_file(self, file_path: Path):
        """
        Process a single SAR tile file.
        
        Args:
            file_path: Path to the SAR tile file
        """
        try:
            # Validate file
            if not self._validate_file(file_path):
                return
            
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            # Create metadata sidecar
            self._create_metadata_sidecar(file_path, metadata)
            
            # Move file to work directory
            work_path = self._move_to_work(file_path)
            
            # Add to work queue
            self._add_to_work_queue(work_path, metadata)
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['last_processed'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(f"Successfully processed {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path.name}: {e}")
            self.stats['files_failed'] += 1
    
    def _validate_file(self, file_path: Path) -> bool:
        """
        Validate a SAR tile file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Check file extension
            if file_path.suffix.lower().lstrip('.') not in self.supported_formats:
                self.logger.warning(f"Unsupported format: {file_path.suffix}")
                return False
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                self.logger.warning(f"File too large: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB")
                return False
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                self.logger.warning(f"File not readable: {file_path}")
                return False
            
            # Try to open as image
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    self.logger.warning(f"Could not read image: {file_path}")
                    return False
                
                # Check image dimensions
                height, width = img.shape[:2]
                if height < 64 or width < 64:
                    self.logger.warning(f"Image too small: {width}x{height}")
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Image validation failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a SAR tile file.
        
        Args:
            file_path: Path to the SAR tile file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size_bytes': file_path.stat().st_size,
            'ingest_timestamp': datetime.now(timezone.utc).isoformat(),
            'file_hash': self._calculate_file_hash(file_path),
            'image_info': self._extract_image_info(file_path)
        }
        
        # Try to extract geospatial information if available
        geo_info = self._extract_geospatial_info(file_path)
        if geo_info:
            metadata['geospatial'] = geo_info
        
        return metadata

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of the file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _extract_image_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic image information."""
        try:
            img = cv2.imread(str(file_path))
            if img is not None:
                height, width = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
                
                return {
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'dtype': str(img.dtype),
                    'size_bytes': img.nbytes
                }
        except Exception as e:
            self.logger.warning(f"Failed to extract image info: {e}")
        
        return {}

    def _extract_geospatial_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract geospatial information from the file.
        
        This is a placeholder - in ideal implementation, we would:
        1. Parse GeoTIFF headers
        2. Extract projection information
        3. Get bounding box coordinates
        4. Parse any embedded metadata
        """
        # For now, return None - this will be enhanced later
        return None

    def _create_metadata_sidecar(self, file_path: Path, metadata: Dict[str, Any]):
        """Create a metadata sidecar file."""
        try:
            metadata_file = self.metadata_dir / f"{file_path.stem}.meta.json"
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.debug(f"Created metadata sidecar: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create metadata sidecar: {e}")
    
    def _move_to_work(self, file_path: Path) -> Path:
        """Move file to the work directory."""
        try:
            work_path = self.work_dir / file_path.name
            
            # Use atomic move if possible
            if hasattr(os, 'replace'):
                os.replace(file_path, work_path)
            else:
                # Fallback for older Python versions
                import shutil
                shutil.move(str(file_path), str(work_path))
            
            self.logger.debug(f"Moved {file_path.name} to work directory")
            return work_path
            
        except Exception as e:
            self.logger.error(f"Failed to move file to work directory: {e}")
            raise
    
    def _add_to_work_queue(self, work_path: Path, metadata: Dict[str, Any]):
        """Add file to the work queue for preprocessing."""
        with self.queue_lock:
            queue_item = {
                'file_path': str(work_path),
                'metadata': metadata,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'pending'
            }
            
            self.work_queue.append(queue_item)
            self.stats['queue_length'] = len(self.work_queue)
            
            self.logger.debug(f"Added {work_path.name} to work queue")
    
    def get_work_item(self) -> Optional[Dict[str, Any]]:
        """
        Get the next work item from the queue.
        
        Returns:
            Next work item or None if queue is empty
        """
        with self.queue_lock:
            if self.work_queue:
                item = self.work_queue.pop(0)
                item['status'] = 'processing'
                self.stats['queue_length'] = len(self.work_queue)
                return item
            return None
    
    def mark_work_complete(self, work_item: Dict[str, Any]):
        """Mark a work item as completed."""
        with self.queue_lock:
            work_item['status'] = 'completed'
            work_item['completion_timestamp'] = datetime.now(timezone.utc).isoformat()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the ingest service."""
        return {
            'service': 'SARIngest',
            'status': 'running' if self.observer.is_alive() else 'stopped',
            'ingest_directory': str(self.ingest_dir),
            'work_directory': str(self.work_dir),
            'statistics': self.stats.copy(),
            'queue_length': len(self.work_queue),
            'supported_formats': self.supported_formats,
            'max_file_size_mb': self.max_file_size_mb
        }


class SARFileHandler(FileSystemEventHandler):
    """
    File handler for SAR ingest service.
    
    This class:
    1. Watches for new files in the ingest directory
    2. Validates SAR tile format and integrity
    3. Creates metadata sidecars (.meta.json) with geospatial info
    4. Moves validated tiles to /work/ready for processing
    """
    def __init__(self, ingest_service: SARIngestService):
        self.ingest_service = ingest_service
        self.logger = logging.getLogger('SARFileHandler')
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self.logger.info(f"New file detected: {file_path.name}")
            
            # Wait a bit to ensure file is fully written
            time.sleep(1)
            
            # Process the file
            self.ingest_service._process_file(file_path)
    
    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory:
            file_path = Path(event.dest_path)
            self.logger.info(f"File moved to ingest: {file_path.name}")
            
            # Process the file
            self.ingest_service._process_file(file_path)


# Import threading at module level
import threading