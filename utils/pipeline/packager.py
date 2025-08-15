"""
Packager Service - Creates downlink packets for satellite transmission
Author: @amanarora9848 (Aman Arora)
"""

from locale import currency
import os
import json
import gzip
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timezone
import time
import zipfile
from datetime import timedelta

from utils.general import file_age

class PackagerService:
    """
    Creates and manages downlink packets for satellite transmission.
    
    This service:
    1. Batches detection results and thumbnails
    2. Compresses data into transmission packets
    3. Adds checksums and metadata
    4. Manages retry queues and outbox
    5. Handles packet prioritization
    """

    def __init__(self, 
                 input_dir: str = "test_postprocessed",
                 thumbs_dir: str = "test_thumbs",
                 outbox_dir: str = "test_outbox",
                 packet_size_mb: float = 10.0,
                 max_packets: int = 100,
                 compression_level: int = 6,
                 max_retries: int = 3,
                 retry_delay_sec: int = 60):
        """
        Initialize the packager service.
        
        Args:
            input_dir: Directory containing post-processed results
            thumbs_dir: Directory containing thumbnail images
            outbox_dir: Directory for outgoing packets
            packet_size_mb: Maximum packet size in MB
            max_packets: Maximum number of packets to keep in outbox
            compression_level: Gzip compression level (1-9)
            max_retries: Maximum retry attempts for failed packets
            retry_delay_sec: Delay between retry attempts
            logger: Logger instance for logging the packager service information
        """

        self.input_dir = Path(input_dir)
        self.thumbs_dir = Path(thumbs_dir)
        self.outbox_dir = Path(outbox_dir)
        self.packet_size_mb = packet_size_mb
        self.max_packets = max_packets
        self.compression_level = compression_level
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        
        # Create outbox directory
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Packet management
        self.packet_counter = 0
        self.retry_queue = []
        self.transmitted_packets = []
        
        # Statistics
        self.stats = {
            'packets_created': 0,
            'packets_transmitted': 0,
            'packets_failed': 0,
            'total_data_mb': 0.0,
            'last_packet_time': None
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the packager service."""
        logger = logging.getLogger('PackagerService')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def create_packets(self, force_create: bool = False) -> List[Dict[str, Any]]:
        """
        Create downlink packets from available data.
        
        Args:
            force_create: Force packet creation even if data is small
            
        Returns:
            List of created packet information
        """

        try:
            # Find available data
            postproc_files = self._find_postprocessed_files()
            thumbnail_files = self._find_thumbnail_files()
            
            if not postproc_files and not thumbnail_files:
                self.logger.info("No data available for packet creation")
                return []
            
            self.logger.info(f"Found {len(postproc_files)} post-processed files and {len(thumbnail_files)} thumbnails")
            
            # Create packets
            packets = []
            current_packet = self._create_new_packet()
            
            # Add post-processed results
            for postproc_file in postproc_files:
                if self._should_start_new_packet(current_packet, postproc_file):
                    if current_packet['files']:  # Don't save empty packets
                        packets.append(self._finalize_packet(current_packet))
                    current_packet = self._create_new_packet()
                
                self._add_file_to_packet(current_packet, postproc_file, 'postprocessed')
            
           # Add thumbnails
            for thumb_file in thumbnail_files:
                if self._should_start_new_packet(current_packet, thumb_file):
                    if current_packet['files']:  # Don't save empty packets
                        packets.append(self._finalize_packet(current_packet))
                    current_packet = self._create_new_packet()
                
                self._add_file_to_packet(current_packet, thumb_file, 'thumbnail')
            
            # Finalize last packet
            if current_packet['files']:
                packets.append(self._finalize_packet(current_packet))
            
            # Save packets to outbox
            for packet in packets:
                self._save_packet(packet)
            
            self.logger.info(f"Created {len(packets)} packets")
            return packets
            
        except Exception as e:
            self.logger.error(f"Failed to create packets: {e}")
            return []
     
    def _find_postprocessed_files(self) -> List[Path]:
        """Find post-processed result files."""
        files = []
        if self.input_dir.exists():
            files.extend(self.input_dir.glob("*.json"))
        return sorted(files)

    def _find_thumbnail_files(self) -> List[Path]:
        """Find thumbnail image files."""
        files = []
        if self.thumbs_dir.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                files.extend(self.thumbs_dir.glob(f"*{ext}"))
        return sorted(files)

    def _create_new_packet(self) -> Dict[str, Any]:
        """Create a new empty packet."""
        self.packet_counter += 1
        timestamp = datetime.now(timezone.utc)
        
        return {
            'packet_id': f"packet_{self.packet_counter:06d}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp_utc': timestamp.isoformat(),
            'files': [],
            'total_size_bytes': 0,
            'file_count': 0,
            'status': 'creating',
            'retry_count': 0,
            'checksum': None,
            'compressed_size_bytes': 0
        }

    def _should_start_new_packet(self, packet: Dict[str, Any], new_file: Path) -> bool:
        """Check if we should start a new packet."""
        if not packet['files']:
            return False
        
        # Check if adding this file would exceed packet size
        file_size = new_file.stat().st_size
        estimated_packet_size = packet['total_size_bytes'] + file_size
        
        # Convert to MB for comparison
        estimated_packet_size_mb = estimated_packet_size / (1024 * 1024)
        
        return estimated_packet_size_mb > self.packet_size_mb
    
    def _add_file_to_packet(self, packet: Dict[str, Any], file_path: Path, file_type: str):
        """Add a file to the current packet."""
        try:
            file_size = file_path.stat().st_size
            file_info = {
                'path': str(file_path),
                'name': file_path.name,
                'size_bytes': file_size,
                'type': file_type,
                'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
            }
            
            packet['files'].append(file_info)
            packet['total_size_bytes'] += file_size
            packet['file_count'] += 1
            
            self.logger.debug(f"Added {file_path.name} to packet {packet['packet_id']}")
            
        except Exception as e:
            self.logger.error(f"Error adding file {file_path} to packet: {e}")

    def _finalize_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize a packet with metadata."""
        packet['status'] = 'ready'
        packet['created_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate packet hash
        packet['checksum'] = self._calculate_packet_checksum(packet)
        
        self.logger.info(f"Finalized packet {packet['packet_id']} with {packet['file_count']} files, "
                        f"size: {packet['total_size_bytes'] / (1024*1024):.2f} MB")
        
        return packet

    def _calculate_packet_checksum(self, packet: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum for packet contents."""
        try:
            # Create a deterministic representation of the packet
            packet_data = {
                'packet_id': packet['packet_id'],
                'timestamp': packet['timestamp_utc'],
                'files': sorted(packet['files'], key=lambda x: x['name'])
            }
            
            packet_json = json.dumps(packet_data, sort_keys=True, default=str)
            return hashlib.sha256(packet_json.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error calculating packet checksum: {e}")
            return "unknown"

    def _save_packet(self, packet: Dict[str, Any]) -> bool:
        """Save packet to outbox directory."""
        try:
            # Create packet directory
            packet_dir = self.outbox_dir / packet['packet_id']
            packet_dir.mkdir(exist_ok=True)
            
            # Copy files to packet directory
            for file_info in packet['files']:
                src_path = Path(file_info['path'])
                dst_path = packet_dir / file_info['name']
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                else:
                    self.logger.warning(f"Source file not found: {src_path}")
            
            # Create packet manifest
            manifest_path = packet_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(packet, f, indent=2, default=str)
            
            # Create compressed archive
            archive_path = self.outbox_dir / f"{packet['packet_id']}.zip"
            self._create_compressed_archive(packet_dir, archive_path)
            
            # Update packet with archive info
            packet['archive_path'] = str(archive_path)
            packet['compressed_size_bytes'] = archive_path.stat().st_size
            
            # Clean up packet directory
            shutil.rmtree(packet_dir)
            
            # Update statistics
            self.stats['packets_created'] += 1
            self.stats['total_data_mb'] += packet['total_size_bytes'] / (1024 * 1024)
            self.stats['last_packet_time'] = datetime.now(timezone.utc).isoformat()
            
            self.logger.info(f"Saved packet {packet['packet_id']} to {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save packet {packet['packet_id']}: {e}")
            return False

    def _create_compressed_archive(self, source_dir: Path, archive_path: Path):
        """Create a compressed ZIP archive of the packet."""
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=self.compression_level) as zipf:
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file():
                        # Add file to archive with relative path
                        arcname = file_path.relative_to(source_dir)
                        zipf.write(file_path, arcname)
            
            self.logger.debug(f"Created compressed archive: {archive_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create compressed archive: {e}")
            raise

    def transmit_packet(self, packet_id: str) -> bool:
        """
        Simulate packet transmission (in trial implementation, this would be send over)
        Args:
            packet_id: ID of the packet to transmit
            
        Returns:
            True if transmission successful, False otherwise
        """
        try:
            # Find packet file
            packet_file = self.outbox_dir / f"{packet_id}.zip"
            if not packet_file.exists():
                self.logger.error(f"Packet file not found: {packet_file}")
                return False
            
            # Simulate transmission delay
            time.sleep(0.1)  # Simulate radio transmission time
            
            # Simulate transmission success/failure (90% success rate for testing)
            import random
            success = random.random() > 0.1
            
            if success:
                # Move to transmitted directory
                transmitted_dir = self.outbox_dir / "transmitted"
                transmitted_dir.mkdir(exist_ok=True)
                
                transmitted_path = transmitted_dir / f"{packet_id}.zip"
                shutil.move(packet_file, transmitted_path)
                
                # Update packet status
                self._update_packet_status(packet_id, 'transmitted')
                
                # Update statistics
                self.stats['packets_transmitted'] += 1
                
                self.logger.info(f"Successfully transmitted packet {packet_id}")
                return True
            else:
                # Simulate transmission failure
                self._handle_transmission_failure(packet_id)
                self.logger.warning(f"Transmission failed for packet {packet_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error transmitting packet {packet_id}: {e}")
            return False

    def _update_packet_status(self, packet_id: str, status: str):
        """Update packet status in manifest."""
        try:
            # Find and update manifest
            manifest_pattern = f"{packet_id}*.json"
            manifest_files = list(self.outbox_dir.glob(manifest_pattern))
            
            if manifest_files:
                manifest_path = manifest_files[0]
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                manifest['status'] = status
                manifest['transmitted_timestamp'] = datetime.now(timezone.utc).isoformat()
                
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"Error updating packet status: {e}")

    def _handle_transmission_failure(self, packet_id: str):
        """Handle transmission failure and manage retry logic."""
        try:
            # Find packet manifest
            manifest_pattern = f"{packet_id}*.json"
            manifest_files = list(self.outbox_dir.glob(manifest_pattern))
            
            if manifest_files:
                manifest_path = manifest_files[0]
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                manifest['retry_count'] = manifest.get('retry_count', 0) + 1
                manifest['last_failure'] = datetime.now(timezone.utc).isoformat()
                
                if manifest['retry_count'] <= self.max_retries:
                    manifest['status'] = 'retry_pending'
                    manifest['next_retry'] = (datetime.now(timezone.utc) + 
                                            timedelta(seconds=self.retry_delay_sec)).isoformat()
                    
                    # Add to retry queue
                    self.retry_queue.append({
                        'packet_id': packet_id,
                        'retry_count': manifest['retry_count'],
                        'next_retry': manifest['next_retry']
                    })
                    
                    self.logger.info(f"Packet {packet_id} queued for retry (attempt {manifest['retry_count']})")
                else:
                    manifest['status'] = 'failed'
                    self.stats['packets_failed'] += 1
                    self.logger.error(f"Packet {packet_id} failed after {manifest['retry_count']} retries")
                
                # Update manifest
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"Error handling transmission failure: {e}")

    def process_retry_queue(self):
        """Process packets in the retry queue."""
        try:
            current_time = datetime.now(timezone.utc)
            ready_retries = []
            
            # Find packets ready for retry
            for retry_item in self.retry_queue:
                next_retry = datetime.fromisoformat(retry_item['next_retry'])
                if current_time >= next_retry:
                    ready_retries.append(retry_item)
            
            # Process ready retries
            for retry_item in ready_retries:
                self.logger.info(f"Retrying packet {retry_item['packet_id']}")
                success = self.transmit_packet(retry_item['packet_id'])
                
                if success:
                    # Remove from retry queue
                    self.retry_queue.remove(retry_item)
                else:
                    # Update retry count and next retry time
                    retry_item['retry_count'] += 1
                    if retry_item['retry_count'] <= self.max_retries:
                        retry_item['next_retry'] = (current_time + 
                                                  timedelta(seconds=self.retry_delay_sec)).isoformat()
                    else:
                        # Remove from retry queue if max retries exceeded
                        self.retry_queue.remove(retry_item)
            
            if ready_retries:
                self.logger.info(f"Processed {len(ready_retries)} retry attempts")
                
        except Exception as e:
            self.logger.error(f"Error processing retry queue: {e}")

    def cleanup_old_packets(self, max_age_hours: int = 24):
        """Clean up old transmitted packets to manage storage."""
        try:
            current_time = datetime.now(timezone.utc)
            transmitted_dir = self.outbox_dir / "transmitted"
            
            if not transmitted_dir.exists():
                return

            cleaned_count = 0
            for packet_file in transmitted_dir.glob("*.zip"):
                # Check file age
                file_age = current_time - datetime.fromtimestamp(packet_file.stat().st_mtime, tz=timezone.utc)
                if file_age.total_seconds() > (max_age_hours * 3600):
                    packet_file.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old packets")

        except Exception as e:
            self.logger.error(f"Error cleaning up old packets: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the packager service."""
        return {
            'service': 'PackagerService',
            'status': 'running',
            'input_directory': str(self.input_dir),
            'thumbnails_directory': str(self.thumbs_dir),
            'outbox_directory': str(self.outbox_dir),
            'packet_parameters': {
                'max_size_mb': self.packet_size_mb,
                'max_packets': self.max_packets,
                'compression_level': self.compression_level,
                'max_retries': self.max_retries
            },
            'queue_status': {
                'retry_queue_length': len(self.retry_queue),
                'packets_in_outbox': len(list(self.outbox_dir.glob("*.zip")))
            },
            'statistics': self.stats.copy()
        }

