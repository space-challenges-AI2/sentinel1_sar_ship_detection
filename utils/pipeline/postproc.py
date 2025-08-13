"""
Post-Processing Service - Handles thumbnail generation and business logic
Author: @amanarora9848 (Aman Arora)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime, timezone
import shutil

class PostProcessingService:
    """
    Post-processes YOLO detection results and generates thumbnails.
    
    This service:
    1. Reads YOLO detection outputs (with NMS already applied)
    2. Generates thumbnails around detections
    3. Applies business logic filters (size, confidence)
    4. Prepares final output for the packager
    """
    
    def __init__(self, 
                 detections_dir: str = "test_detections",
                 output_dir: str = "test_postprocessed",
                 thumbs_dir: str = "test_thumbs",
                 min_confidence: float = 0.5,
                 min_detection_size: int = 32,
                 thumbnail_size: int = 256,
                 save_crops: bool = True):
        """
        Initialize the post-processing service.
        
        Args:
            detections_dir: Directory containing YOLO detection results
            output_dir: Directory for post-processed outputs
            thumbs_dir: Directory for thumbnail images
            min_confidence: Minimum confidence threshold
            min_detection_size: Minimum detection size in pixels
            thumbnail_size: Size of generated thumbnails
            save_crops: Whether to save cropped thumbnails
        """
        self.detections_dir = Path(detections_dir)
        self.output_dir = Path(output_dir)
        self.thumbs_dir = Path(thumbs_dir)
        self.min_confidence = min_confidence
        self.min_detection_size = min_detection_size
        self.thumbnail_size = thumbnail_size
        self.save_crops = save_crops
        
        # Create output directories
        for dir_path in [self.output_dir, self.thumbs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Statistics
        self.stats = {
            'detections_processed': 0,
            'thumbnails_generated': 0,
            'detections_filtered': 0,
            'last_processed': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the post-processing service."""
        logger = logging.getLogger('PostProcessingService')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_detections(self, detection_results_dir: str, original_images_dir: str = None) -> Optional[Dict[str, Any]]:
        """
        Process detection results and generate thumbnails.
        
        Args:
            detection_results_dir: Directory containing YOLO detection results
            original_images_dir: Directory containing original input images
            
        Returns:
            Post-processing results or None if failed
        """
        try:
            detection_dir = Path(detection_results_dir)
            if not detection_dir.exists():
                self.logger.error(f"Detection directory not found: {detection_dir}")
                return None
            
            # Find detection results
            detection_files = self._find_detection_files(detection_dir)
            if not detection_files:
                self.logger.warning(f"No detection files found in {detection_dir}")
                return None
            
            # Process each detection file
            processed_results = []
            for detection_file in detection_files:
                result = self._process_single_detection(detection_file, original_images_dir)
                if result:
                    processed_results.append(result)
            
            # Create final output
            output_data = {
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'service': 'PostProcessingService',
                'total_detections': len(processed_results),
                'processing_params': {
                    'min_confidence': self.min_confidence,
                    'min_detection_size': self.min_detection_size,
                    'thumbnail_size': self.thumbnail_size
                },
                'results': processed_results
            }
            
            # Save output
            output_path = self._save_output(output_data)
            
            # Update statistics
            self.stats['detections_processed'] += len(processed_results)
            self.stats['last_processed'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(f"Processed {len(processed_results)} detections, saved to {output_path}")
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'detections_count': len(processed_results),
                'thumbnails_count': self.stats['thumbnails_generated'],
                'output_data': output_data
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process detections: {e}")
            return None
    
    def _find_detection_files(self, detection_dir: Path) -> List[Path]:
        """Find all detection result files."""
        detection_files = []
        
        # Look for labels directory (YOLO text format) - check both root and 'pipeline' subdirectory
        possible_labels_dirs = [
            detection_dir / "labels",
            detection_dir / "pipeline" / "labels"
        ]
        
        for labels_dir in possible_labels_dirs:
            if labels_dir.exists():
                self.logger.info(f"Found labels directory: {labels_dir}")
                detection_files.extend(labels_dir.glob("*.txt"))
        
        # Look for JSON results in both root and 'pipeline' subdirectory
        possible_json_dirs = [
            detection_dir,
            detection_dir / "pipeline"
        ]
        
        for json_dir in possible_json_dirs:
            if json_dir.exists():
                detection_files.extend(json_dir.glob("*.json"))
                # Also look for other result files
                detection_files.extend(json_dir.glob("*.txt"))
        
        # Remove duplicates and sort
        detection_files = list(set(detection_files))
        detection_files.sort()
        
        self.logger.info(f"Found {len(detection_files)} detection files: {[f.name for f in detection_files]}")
        
        return detection_files
    
    def _process_single_detection(self, detection_file: Path, original_images_dir: str = None) -> Optional[Dict[str, Any]]:
        """Process a single detection file."""
        try:
            # Load detection results
            detections = self._load_detections(detection_file)
            if not detections:
                return None
            
            # Find corresponding original image
            original_image_path = self._find_original_image(detection_file, original_images_dir)
            if not original_image_path:
                self.logger.warning(f"Original image not found for {detection_file}")
                return None
            
            # Load original image
            original_image = cv2.imread(str(original_image_path))
            if original_image is None:
                self.logger.error(f"Could not load image: {original_image_path}")
                return None
            
            # Process detections
            processed_detections = []
            for detection in detections:
                # Apply business logic filters
                if self._passes_filters(detection):
                    # Generate thumbnail
                    thumbnail_path = self._generate_thumbnail(original_image, detection, detection_file)
                    
                    processed_detection = {
                        'detection': detection,
                        'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                        'filtered': False
                    }
                    
                    processed_detections.append(processed_detection)
                else:
                    # Mark as filtered out
                    processed_detections.append({
                        'detection': detection,
                        'thumbnail_path': None,
                        'filtered': True
                    })
                    self.stats['detections_filtered'] += 1
            
            return {
                'detection_file': str(detection_file),
                'original_image': str(original_image_path),
                'detections': processed_detections,
                'total_detections': len(detections),
                'passed_filters': len([d for d in processed_detections if not d['filtered']])
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {detection_file}: {e}")
            return None
    
    def _load_detections(self, detection_file: Path) -> List[Dict[str, Any]]:
        """Load detection results from file."""
        try:
            if detection_file.suffix == '.txt':
                # YOLO text format: class x_center y_center width height confidence
                detections = []
                with open(detection_file, 'r') as f:
                    for line_num, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            detection = {
                                'class': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4]),
                                'confidence': float(parts[5]),
                                'line_number': line_num
                            }
                            detections.append(detection)
                return detections
            
            elif detection_file.suffix == '.json':
                # JSON format
                with open(detection_file, 'r') as f:
                    data = json.load(f)
                    if 'detections' in data:
                        return data['detections']
                    elif 'results' in data:
                        return data['results']
                    else:
                        return data
            
            else:
                self.logger.warning(f"Unsupported detection file format: {detection_file.suffix}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading detections from {detection_file}: {e}")
            return []
    
    def _find_original_image(self, detection_file: Path, original_images_dir: str = None) -> Optional[Path]:
        """Find the original image corresponding to detection results."""
        try:
            # Get base name without extension
            base_name = detection_file.stem
            
            # Look in original images directory if provided
            if original_images_dir:
                original_dir = Path(original_images_dir)
                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    image_path = original_dir / f"{base_name}{ext}"
                    if image_path.exists():
                        return image_path
            
            # Look in common locations
            common_locations = [
                Path("test_ingest"),
                Path("test_work"),
                Path("source")
            ]
            
            for location in common_locations:
                if location.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        image_path = location / f"{base_name}{ext}"
                        if image_path.exists():
                            return image_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding original image: {e}")
            return None
    
    def _passes_filters(self, detection: Dict[str, Any]) -> bool:
        """Check if detection passes business logic filters."""
        try:
            # Confidence threshold
            if detection.get('confidence', 0) < self.min_confidence:
                return False
            
            # Size threshold (if we have width/height)
            if 'width' in detection and 'height' in detection:
                # Convert normalized coordinates to pixels (assuming 640x640 input)
                width_px = detection['width'] * 640
                height_px = detection['height'] * 640
                
                if width_px < self.min_detection_size or height_px < self.min_detection_size:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in filter check: {e}")
            return False
    
    def _generate_thumbnail(self, image: np.ndarray, detection: Dict[str, Any], detection_file: Path) -> Optional[Path]:
        """Generate thumbnail around detection."""
        try:
            if not self.save_crops:
                return None
            
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Convert YOLO normalized coordinates to pixel coordinates
            x_center = detection['x_center'] * img_width
            y_center = detection['y_center'] * img_height
            width = detection['width'] * img_width
            height = detection['height'] * img_height
            
            # Calculate bounding box
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_width, int(x_center + width / 2))
            y2 = min(img_height, int(y_center + height / 2))
            
            # Add padding for thumbnail
            padding = self.thumbnail_size // 4
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_width, x2 + padding)
            y2 = min(img_height, y2 + padding)
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Resize to thumbnail size
            thumbnail = cv2.resize(region, (self.thumbnail_size, self.thumbnail_size))
            
            # Save thumbnail
            base_name = detection_file.stem
            thumbnail_name = f"{base_name}_det_{detection.get('line_number', 0)}_thumb.jpg"
            thumbnail_path = self.thumbs_dir / thumbnail_name
            
            cv2.imwrite(str(thumbnail_path), thumbnail)
            
            self.stats['thumbnails_generated'] += 1
            self.logger.debug(f"Generated thumbnail: {thumbnail_path}")
            
            return thumbnail_path
            
        except Exception as e:
            self.logger.error(f"Error generating thumbnail: {e}")
            return None
    
    def _save_output(self, output_data: Dict[str, Any]) -> Path:
        """Save post-processing output."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"postprocessed_{timestamp}.json"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.logger.debug(f"Saved output to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving output: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the post-processing service."""
        return {
            'service': 'PostProcessingService',
            'status': 'running',
            'detections_directory': str(self.detections_dir),
            'output_directory': str(self.output_dir),
            'thumbnails_directory': str(self.thumbs_dir),
            'processing_parameters': {
                'min_confidence': self.min_confidence,
                'min_detection_size': self.min_detection_size,
                'thumbnail_size': self.thumbnail_size,
                'save_crops': self.save_crops
            },
            'statistics': self.stats.copy()
        } 