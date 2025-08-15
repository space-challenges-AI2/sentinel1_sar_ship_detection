"""
Enhanced Health Monitoring Service - Production-ready system monitoring
Author: @amanarora9848 (Aman Arora)
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timezone

# Optional imports with fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available - system monitoring disabled")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("Warning: GPUtil not available - GPU monitoring limited")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available - CUDA monitoring disabled")

class HealthMonitor:  # Changed back to HealthMonitor for compatibility
    """
    Production-ready system health monitor with GPU, CPU, and AI model monitoring.
    
    This service monitors:
    1. GPU memory and utilization (CUDA) - if available
    2. System resources (CPU, RAM, disk, network) - if available
    3. AI model performance metrics
    4. Pipeline health and telemetry
    5. Alert thresholds and notifications
    """
    
    def __init__(self, 
                 pipeline_dir: Path,
                 logs_dir: Path,
                 alert_thresholds: Dict[str, float] = None):
        """
        Initialize enhanced health monitor.
        
        Args:
            pipeline_dir: Root pipeline directory
            logs_dir: Logs directory path
            alert_thresholds: Custom alert thresholds
        """
        self.pipeline_dir = Path(pipeline_dir)
        self.logs_dir = Path(logs_dir)
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'gpu_memory_usage': 0.9,      # 90% GPU memory
            'gpu_temperature': 85.0,       # 85°C GPU temp
            'cpu_usage': 0.95,             # 95% CPU usage
            'ram_usage': 0.9,              # 90% RAM usage
            'disk_usage': 0.95,            # 95% disk usage
            'inference_latency': 10.0      # 10 seconds max
        }
        
        # Ensure logs directory exists
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Performance tracking
        self.start_time = time.time()
        self.metrics = {
            'total_images_processed': 0,
            'total_detections': 0,
            'total_processing_time': 0.0,
            'errors': [],
            'last_heartbeat': time.time(),
            'alerts': [],
            'system_health': 'healthy'
        }
        
        # Health monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Check system capabilities
        self._check_system_capabilities()
        
        self.logger.info("Enhanced health monitor initialized")
    
    def _check_system_capabilities(self):
        """Check what monitoring capabilities are available."""
        self.has_gpu = HAS_TORCH and torch.cuda.is_available()
        self.has_psutil = HAS_PSUTIL
        self.has_gputil = HAS_GPUTIL
        
        if self.has_gpu:
            self.logger.info(f"GPU monitoring enabled: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("GPU monitoring disabled - no CUDA available")
        
        if self.has_psutil:
            self.logger.info("System monitoring enabled")
        else:
            self.logger.info("System monitoring disabled - psutil not available")
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics including memory, utilization, and temperature."""
        if not self.has_gpu:
            return {'status': 'no_gpu', 'message': 'CUDA not available'}
        
        try:
            # PyTorch CUDA metrics
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            # GPUtil for additional metrics (if available)
            gpu_utilization = 0
            gpu_temperature = 0
            gpu_memory_used = 0
            gpu_memory_total_gputil = 0
            
            if self.has_gputil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Primary GPU
                        gpu_utilization = gpu.load * 100
                        gpu_temperature = gpu.temperature
                        gpu_memory_used = gpu.memoryUsed
                        gpu_memory_total_gputil = gpu.memoryTotal
                except Exception as e:
                    self.logger.debug(f"GPUtil metrics failed: {e}")
            
            return {
                'status': 'active',
                'memory_allocated_gb': round(gpu_memory_allocated, 2),
                'memory_reserved_gb': round(gpu_memory_reserved, 2),
                'memory_total_gb': round(gpu_memory_total, 2),
                'memory_usage_percent': round((gpu_memory_allocated / gpu_memory_total) * 100, 1),
                'utilization_percent': round(gpu_utilization, 1),
                'temperature_celsius': gpu_temperature,
                'memory_used_mb': gpu_memory_used,
                'memory_total_mb': gpu_memory_total_gputil
            }
            
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        if not self.has_psutil:
            return {'status': 'no_psutil', 'message': 'psutil not available'}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'count': cpu_count,
                    'frequency_mhz': cpu_freq.current if cpu_freq else 0
                },
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'memory_available_gb': round(memory.available / (1024**3), 2),
                    'memory_used_gb': round(memory.used / (1024**3), 2),
                    'usage_percent': memory.percent,
                    'swap_total_gb': round(swap.total / (1024**3), 2),
                    'swap_used_gb': round(swap.used / (1024**3), 2)
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'usage_percent': round((disk.used / disk.total) * 100, 1)
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_alerts(self, gpu_metrics: Dict, system_metrics: Dict):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        # GPU alerts
        if gpu_metrics.get('status') == 'active':
            if gpu_metrics.get('memory_usage_percent', 0) > self.alert_thresholds['gpu_memory_usage'] * 100:
                alerts.append({
                    'level': 'warning',
                    'type': 'gpu_memory',
                    'message': f"GPU memory usage high: {gpu_metrics['memory_usage_percent']}%",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            if gpu_metrics.get('temperature_celsius', 0) > self.alert_thresholds['gpu_temperature']:
                alerts.append({
                    'level': 'critical',
                    'type': 'gpu_temperature',
                    'message': f"GPU temperature critical: {gpu_metrics['temperature_celsius']}°C",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        # System alerts (only if psutil is available)
        if system_metrics.get('status') != 'no_psutil':
            if system_metrics.get('cpu', {}).get('usage_percent', 0) > self.alert_thresholds['cpu_usage'] * 100:
                alerts.append({
                    'level': 'warning',
                    'type': 'cpu_usage',
                    'message': f"CPU usage high: {system_metrics['cpu']['usage_percent']}%",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            if system_metrics.get('memory', {}).get('usage_percent', 0) > self.alert_thresholds['ram_usage'] * 100:
                alerts.append({
                    'level': 'warning',
                    'type': 'ram_usage',
                    'message': f"RAM usage high: {system_metrics['memory']['usage_percent']}%",
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        return alerts
    
    def _collect_metrics(self):
        """Collect comprehensive system metrics."""
        try:
            # Get GPU and system metrics
            gpu_metrics = self._get_gpu_metrics()
            system_metrics = self._get_system_metrics()
            
            # Check for alerts
            alerts = self._check_alerts(gpu_metrics, system_metrics)
            
            # Update metrics
            self.metrics['last_heartbeat'] = time.time()
            self.metrics['gpu_metrics'] = gpu_metrics
            self.metrics['system_metrics'] = system_metrics
            self.metrics['alerts'].extend(alerts)
            
            # Keep only last 100 alerts
            if len(self.metrics['alerts']) > 100:
                self.metrics['alerts'] = self.metrics['alerts'][-100:]
            
            # Update system health status
            if any(alert['level'] == 'critical' for alert in alerts):
                self.metrics['system_health'] = 'critical'
            elif any(alert['level'] == 'warning' for alert in alerts):
                self.metrics['system_health'] = 'warning'
            else:
                self.metrics['system_health'] = 'healthy'
            
            # Log any alerts
            for alert in alerts:
                if alert['level'] == 'critical':
                    self.logger.critical(alert['message'])
                elif alert['level'] == 'warning':
                    self.logger.warning(alert['message'])
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def _log_current_metrics(self):
        """Log a summary of current metrics."""
        try:
            gpu_metrics = self.metrics.get('gpu_metrics', {})
            system_metrics = self.metrics.get('system_metrics', {})
            
            # Log GPU metrics
            if gpu_metrics.get('status') == 'active':
                self.logger.info(f"GPU Status: {gpu_metrics.get('memory_usage_percent', 0)}% memory, "
                               f"{gpu_metrics.get('utilization_percent', 0)}% utilization, "
                               f"{gpu_metrics.get('temperature_celsius', 0)}°C")
            elif gpu_metrics.get('status') == 'no_gpu':
                self.logger.info("GPU Status: No CUDA GPU available")
            else:
                self.logger.info(f"GPU Status: {gpu_metrics.get('status', 'unknown')}")
            
            # Log system metrics
            if system_metrics.get('status') != 'no_psutil':
                cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
                ram_usage = system_metrics.get('memory', {}).get('usage_percent', 0)
                disk_usage = system_metrics.get('disk', {}).get('usage_percent', 0)
                
                self.logger.info(f"System Status: CPU {cpu_usage}%, RAM {ram_usage}%, Disk {disk_usage}%")
            else:
                self.logger.info("System Status: psutil not available")
                
        except Exception as e:
            self.logger.error(f"Error logging metrics summary: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        uptime = time.time() - self.start_time
        
        status = {
            'status': self.metrics['system_health'],
            'uptime_seconds': uptime,
            'uptime_human': self._format_uptime(uptime),
            'metrics': self.metrics.copy(),
            'gpu_status': self.metrics.get('gpu_metrics', {}),
            'system_status': self.metrics.get('system_metrics', {}),
            'alerts': self.metrics.get('alerts', []),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return status
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def record_processing(self, image_count: int, detection_count: int, processing_time: float):
        """Record processing metrics."""
        self.metrics['total_images_processed'] += image_count
        self.metrics['total_detections'] += detection_count
        self.metrics['total_processing_time'] += processing_time
        
        # Calculate average processing time
        if self.metrics['total_images_processed'] > 0:
            avg_time = self.metrics['total_processing_time'] / self.metrics['total_images_processed']
            self.metrics['avg_processing_time'] = avg_time
    
    def record_error(self, error: str):
        """Record an error."""
        self.metrics['errors'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': error
        })
        
        # Keep only last 100 errors
        if len(self.metrics['errors']) > 100:
            self.metrics['errors'] = self.metrics['errors'][-100:]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for health monitoring."""
        logger = logging.getLogger('HealthMonitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = self.logs_dir / f"health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def start_monitoring(self):
        """Start health monitoring thread."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Health monitoring started")
            
            # Collect initial metrics immediately
            self.logger.info("Collecting initial system metrics...")
            self._collect_metrics()
            self._log_current_metrics()
    
    def stop_monitoring(self):
        """Stop health monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._collect_metrics()
                # Use shorter interval for more responsive monitoring
                time.sleep(2)  # Check every 10 seconds instead of 30
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error