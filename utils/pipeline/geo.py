"""
Georeferencing Service - Converts pixel coordinates to geographic coordinates
Author: @amanarora9848 (Aman Arora)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timezone
import math

class GeoreferencingService:
    """
    Converts YOLO detection bounding boxes to geographic coordinates.
    
    This service:
    1. Reads metadata sidecars (.meta.json) from the ingest stage
    2. Converts pixel bboxes to geographic coordinates
    3. Generates WKT (Well-Known Text) footprints
    4. Provides coordinate transformation utilities
    """
    
    def __init__(self, 
                 metadata_dir: str = "/work/metadata",
                 detections_dir: str = "/detections",
                 output_dir: str = "/work/georeferenced"):
        """
        Initialize the georeferencing service.
        
        Args:
            metadata_dir: Directory containing metadata sidecars
            detections_dir: Directory containing YOLO detection results
            output_dir: Directory for georeferenced outputs
        """
        self.metadata_dir = Path(metadata_dir)
        self.detections_dir = Path(detections_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Statistics
        self.stats = {
            'detections_processed': 0,
            'detections_failed': 0,
            'last_processed': None
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the georeferencing service."""
        logger = logging.getLogger('GeoreferencingService')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def process_detections(self, detection_file: str, metadata_file: str = None) -> Optional[Dict[str, Any]]:
        """
        Process YOLO detections and convert to geographic coordinates.
        
        Args:
            detection_file: Path to YOLO detection results
            metadata_file: Optional path to metadata sidecar
            
        Returns:
            Georeferenced detection results or None if failed
        """

        try:
            # Load detections
            detections = self._load_detections(detection_file)
            if not detections:
                return None
            
            # Load or find metadata
            if metadata_file is None:
                metadata_file = self._find_metadata_file(detection_file)
            
            metadata = self._load_metadata(metadata_file)
            if not metadata:
                self.logger.warning(f"No metadata found for {detection_file}")
                # Use default metadata for testing
                metadata = self._create_default_metadata(detection_file)
            
            # Process each detection
            georeferenced_detections = []
            for detection in detections:
                geo_detection = self._georeference_detection(detection, metadata)
                if geo_detection:
                    georeferenced_detections.append(geo_detection)

            # Create output
            output_data = {
                'scene_id': metadata.get('scene_id', 'unknown'),
                'tile_id': metadata.get('tile_id', 'unknown'),
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'detections': georeferenced_detections,
                'metadata': metadata,
                'processing_info': {
                    'service': 'GeoreferencingService',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_detections': len(georeferenced_detections)
                }
            }
            
            # Save georeferenced results
            output_path = self._save_georeferenced_results(detection_file, output_data)
            
            # Update statistics
            self.stats['detections_processed'] += len(georeferenced_detections)
            self.stats['last_processed'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(f"Processed {len(georeferenced_detections)} detections from {detection_file}")
            
            return {
                'status': 'success',
                'input_file': detection_file,
                'output_file': str(output_path),
                'detections_count': len(georeferenced_detections),
                'output_data': output_data
            }

        except Exception as e:
            self.logger.error(f"Failed to process detections from {detection_file}: {e}")
            self.stats['detections_failed'] += 1
            return None
    
    def _load_detections(self, detection_file: str) -> List[Dict[str, Any]]:
        """Load YOLO detection results from file."""
        try:
            detection_path = Path(detection_file)
            
            # Handle different detection output formats
            if detection_path.suffix == '.json':
                with open(detection_path, 'r') as f:
                    data = json.load(f)
                    # Extract detections from YOLO output format
                    if 'detections' in data:
                        return data['detections']
                    elif 'results' in data:
                        return data['results']
                    else:
                        return data

            elif detection_path.suffix == '.txt':
                # Parse YOLO text format (class x_center y_center width height confidence)
                detections = []
                with open(detection_path, 'r') as f:
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

            else:
                self.logger.warning(f"Unsupported detection file format: {detection_path.suffix}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading detections from {detection_file}: {e}")
            return []
        
    def _find_metadata_file(self, detection_file: str) -> Optional[str]:
        """Find the corresponding metadata file for a detection file."""
        try:
            detection_path = Path(detection_file)
            base_name = detection_path.stem
            
            # Look for metadata file with same base name
            metadata_file = self.metadata_dir / f"{base_name}.meta.json"
            
            if metadata_file.exists():
                return str(metadata_file)
            
            # Try alternative naming patterns
            alternatives = [
                f"{base_name}_meta.json",
                f"{base_name}.metadata.json",
                f"{base_name}.json"
            ]
            
            for alt_name in alternatives:
                alt_path = self.metadata_dir / alt_name
                if alt_path.exists():
                    return str(alt_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding metadata file: {e}")
            return None

    def _load_metadata(self, metadata_file: str) -> Optional[Dict[str, Any]]:
        """Load metadata from sidecar file."""
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.logger.debug(f"Loaded metadata from {metadata_file}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error loading metadata from {metadata_file}: {e}")
            return None

    def _create_default_metadata(self, detection_file: str) -> Dict[str, Any]:
        """Create default metadata for testing when no sidecar is available."""
        detection_path = Path(detection_file)
        
        # Generate synthetic metadata for testing
        metadata = {
            'filename': detection_path.name,
            'scene_id': f"test_scene_{detection_path.stem}",
            'tile_id': detection_path.stem,
            'ingest_timestamp': datetime.now(timezone.utc).isoformat(),
            'image_info': {
                'width': 640,  # Default YOLO input size
                'height': 640,
                'channels': 3
            },
            'geospatial': {
                'projection': 'WGS84',
                'bounds': {
                    'north': 45.0,
                    'south': 44.0,
                    'east': 25.0,
                    'west': 24.0
                },
                'pixel_size_x': 0.001,  # degrees per pixel
                'pixel_size_y': 0.001
            }
        }
        
        self.logger.info(f"Created default metadata for {detection_file}")
        return metadata

    def _georeference_detection(self, detection: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert a single detection from pixel to geographic coordinates.
        
        Args:
            detection: YOLO detection result
            metadata: Image metadata with geospatial information
            
        Returns:
            Georeferenced detection or None if failed
        """
        try:
            # Extract image dimensions
            image_info = metadata.get('image_info', {})
            img_width = image_info.get('width', 640)
            img_height = image_info.get('height', 640)
            
            # Extract geospatial bounds
            geo_info = metadata.get('geospatial', {})
            bounds = geo_info.get('bounds', {})
            
            # Calculate pixel sizes (degrees per pixel)
            pixel_size_x = geo_info.get('pixel_size_x', 
                (bounds.get('east', 0) - bounds.get('west', 0)) / img_width)
            pixel_size_y = geo_info.get('pixel_size_y',
                (bounds.get('north', 0) - bounds.get('south', 0)) / img_height)
            
            # Convert YOLO coordinates to pixel coordinates
            if 'x_center' in detection and 'y_center' in detection:
                # YOLO format: normalized center coordinates
                x_center = detection['x_center'] * img_width
                y_center = detection['y_center'] * img_height
                width = detection['width'] * img_width
                height = detection['height'] * img_height
            else:
                # Assume already in pixel coordinates
                x_center = detection.get('x_center', 0)
                y_center = detection.get('y_center', 0)
                width = detection.get('width', 0)
                height = detection.get('height', 0)
            
            # Convert to bounding box coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Convert to geographic coordinates
            lon1 = bounds.get('west', 0) + x1 * pixel_size_x
            lat1 = bounds.get('north', 0) - y1 * pixel_size_y
            lon2 = bounds.get('west', 0) + x2 * pixel_size_x
            lat2 = bounds.get('north', 0) - y2 * pixel_size_y
            
            # Calculate centroid
            centroid_lon = (lon1 + lon2) / 2
            centroid_lat = (lat1 + lat2) / 2
            
            # Generate WKT footprint
            wkt_footprint = self._generate_wkt_footprint(lon1, lat1, lon2, lat2)
            
            # Create georeferenced detection
            geo_detection = {
                'bbox_image_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                'bbox_geographic': {
                    'west': lon1,
                    'south': lat1,
                    'east': lon2,
                    'north': lat2
                },
                'centroid': {
                    'lon': centroid_lon,
                    'lat': centroid_lat
                },
                'footprint_wkt': wkt_footprint,
                'confidence': detection.get('confidence', 0.0),
                'class': detection.get('class', 0),
                'class_name': self._get_class_name(detection.get('class', 0)),
                'pixel_coordinates': {
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                }
            }
            
            return geo_detection
            
        except Exception as e:
            self.logger.error(f"Error georeferencing detection: {e}")
            return None

    def _generate_wkt_footprint(self, lon1: float, lat1: float, lon2: float, lat2: float) -> str:
        """Generate WKT (Well-Known Text) polygon from bounding box."""
        # Create a simple rectangular polygon
        wkt = f"POLYGON(({lon1} {lat1}, {lon2} {lat1}, {lon2} {lat2}, {lon1} {lat2}, {lon1} {lat1}))"
        return wkt
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        class_names = {
            0: 'ship',
            1: 'boat',
            2: 'vessel'
        }
        return class_names.get(class_id, f'class_{class_id}')
    
    def _save_georeferenced_results(self, detection_file: str, output_data: Dict[str, Any]) -> Path:
        """Save georeferenced results to output directory."""
        try:
            detection_path = Path(detection_file)
            base_name = detection_path.stem
            
            # Create output filename
            output_filename = f"{base_name}_georeferenced.json"
            output_path = self.output_dir / output_filename
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.logger.debug(f"Saved georeferenced results: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save georeferenced results: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the georeferencing service."""
        return {
            'service': 'GeoreferencingService',
            'status': 'running',
            'metadata_directory': str(self.metadata_dir),
            'detections_directory': str(self.detections_dir),
            'output_directory': str(self.output_dir),
            'statistics': self.stats.copy()
        }
        