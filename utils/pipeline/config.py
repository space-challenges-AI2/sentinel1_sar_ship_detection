"""
Configuration management for the pipeline
"""

import os
from pathlib import Path
from typing import Dict, Any
import logging

class PipelineConfig:
    """Loads and manages pipeline configuration from environment files."""
    
    def __init__(self, env_file: str = "configs/flight.env"):
        self.env_file = Path(env_file)
        self.config = {}
        self.logger = logging.getLogger('PipelineConfig')
        
        if self.env_file.exists():
            self._load_env_file()
        else:
            self.logger.warning(f"Environment file not found: {env_file}")
            self._load_defaults()
        
        # Post-process configuration to handle paths
        self._process_paths()
    
    def _load_env_file(self):
        """Load configuration from .env file."""
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self.config[key.strip()] = value.strip()
            
            self.logger.info(f"Loaded configuration from {self.env_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load environment file: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.config = {
            'INGEST_DIR': 'test_ingest',
            'WORK_DIR': 'test_work',
            'METADATA_DIR': 'test_metadata',
            'DETECTIONS_DIR': 'test_detections',
            'OUTBOX_DIR': 'test_outbox',
            'LOGS_DIR': 'test_logs',
            'PREPROC_MODE': 'fabf',
            'FABF_RHO': '5.0',
            'FABF_N': '5',
            'FABF_SIGMA': '0.1',
            'CONFIDENCE_THRESHOLD': '0.5',
            'IOU_THRESHOLD': '0.45',
            'MAX_DETECTIONS': '1000',
            'BATCH_SIZE': '1',
            'DEVICE': 'auto',
            'LOG_LEVEL': 'INFO'
        }
        self.logger.info("Using default configuration")
    
    def _process_paths(self):
        """Process and validate path configurations."""
        # Get the project root (where the config file is located)
        if self.env_file.exists():
            project_root = self.env_file.parent.parent
        else:
            # Fallback: assume we're in the project root
            project_root = Path.cwd()
        
        # Check if we're in a test environment
        test_ingest = project_root / "utils" / "pipeline" / "test_ingest"
        is_test_environment = test_ingest.exists()
        
        if is_test_environment:
            self.logger.info("Detected test environment, adjusting paths")
            
            # Override absolute paths with test paths
            if self.config.get('MODEL_WEIGHTS', '').startswith('/'):
                # Try to find weights in the project
                possible_weights = [
                    "runs/train/experiment3/weights/best.pt",
                    "weights/best.pt",
                    "runs/train/exp/weights/best.pt"
                ]
                
                for weight_path in possible_weights:
                    if (project_root / weight_path).exists():
                        self.config['MODEL_WEIGHTS'] = str(project_root / weight_path)
                        self.logger.info(f"Found weights at: {self.config['MODEL_WEIGHTS']}")
                        break
                else:
                    # If no weights found, use a default test path
                    self.config['MODEL_WEIGHTS'] = "runs/train/experiment3/weights/best.pt"
                    self.logger.warning(f"No weights found, using default: {self.config['MODEL_WEIGHTS']}")
            
            # Override other absolute paths for testing
            path_overrides = {
                'INGEST_DIR': 'utils/pipeline/test_ingest',
                'WORK_DIR': 'utils/pipeline/test_work',
                'METADATA_DIR': 'utils/pipeline/test_metadata',
                'DETECTIONS_DIR': 'utils/pipeline/test_detections',
                'OUTBOX_DIR': 'utils/pipeline/test_outbox',
                'LOGS_DIR': 'utils/pipeline/test_logs'
            }
            
            for key, test_path in path_overrides.items():
                if self.config.get(key, '').startswith('/'):
                    self.config[key] = test_path
                    self.logger.info(f"Overriding {key} with test path: {test_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        try:
            return int(self.config.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        try:
            return float(self.config.get(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.config.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_path(self, key: str, default: str = None) -> Path:
        """Get configuration value as Path object."""
        value = self.config.get(key, default)
        if value:
            return Path(value)
        return Path(default) if default else Path.cwd() 