"""
Jetson-Compatible Display Handler
Handles display operations on Jetson devices without display server
Author: @amanarora9848
"""

import cv2
import os
import platform
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class JetsonDisplayHandler:
    """Handles display operations on Jetson devices"""
    
    def __init__(self):
        self.is_jetson = self._detect_jetson()
        self.has_display = self._check_display()
        self.display_enabled = self.has_display and not self.is_jetson
        
        if self.is_jetson:
            logger.info("Jetson detected - using headless display mode")
        elif not self.has_display:
            logger.warning("No display detected - using headless mode")
        else:
            logger.info("Display available - using interactive mode")
    
    def _detect_jetson(self):
        """Detect if running on Jetson"""
        try:
            # Check architecture
            if platform.machine() != 'aarch64':
                return False
            
            # Check for Jetson-specific files
            jetson_files = ['/etc/nv_tegra_release', '/sys/module/tegra_fuse/parameters/tegra_chip_id']
            return any(Path(f).exists() for f in jetson_files)
            
        except Exception:
            return False
    
    def _check_display(self):
        """Check if display is available"""
        try:
            # Check DISPLAY environment variable
            display = os.environ.get('DISPLAY')
            if not display:
                return False
            
            # Check if we can actually create a window
            test_img = cv2.imread('/dev/null') if os.path.exists('/dev/null') else None
            if test_img is None:
                test_img = cv2.imread('/tmp/test.png') if os.path.exists('/tmp/test.png') else None
            
            if test_img is not None:
                # Try to create a test window
                cv2.namedWindow('test', cv2.WINDOW_NORMAL)
                cv2.destroyWindow('test')
                return True
            
            return False
            
        except Exception:
            return False
    
    def imshow(self, window_name, image):
        """Safe imshow that works on Jetson"""
        if not self.display_enabled:
            logger.debug(f"Display disabled - skipping imshow for {window_name}")
            return
        
        try:
            cv2.imshow(window_name, image)
        except Exception as e:
            logger.warning(f"imshow failed: {e}")
            self.display_enabled = False
    
    def namedWindow(self, window_name, flags=None):
        """Safe namedWindow that works on Jetson"""
        if not self.display_enabled:
            logger.debug(f"Display disabled - skipping namedWindow for {window_name}")
            return
        
        try:
            if flags is not None:
                cv2.namedWindow(window_name, flags)
            else:
                cv2.namedWindow(window_name)
        except Exception as e:
            logger.warning(f"namedWindow failed: {e}")
            self.display_enabled = False
    
    def resizeWindow(self, window_name, width, height):
        """Safe resizeWindow that works on Jetson"""
        if not self.display_enabled:
            logger.debug(f"Display disabled - skipping resizeWindow for {window_name}")
            return
        
        try:
            cv2.resizeWindow(window_name, width, height)
        except Exception as e:
            logger.warning(f"resizeWindow failed: {e}")
            self.display_enabled = False
    
    def waitKey(self, delay):
        """Safe waitKey that works on Jetson"""
        if not self.display_enabled:
            logger.debug(f"Display disabled - skipping waitKey")
            return -1
        
        try:
            return cv2.waitKey(delay)
        except Exception as e:
            logger.warning(f"waitKey failed: {e}")
            self.display_enabled = False
            return -1
    
    def destroyAllWindows(self):
        """Safe destroyAllWindows that works on Jetson"""
        if not self.display_enabled:
            logger.debug(f"Display disabled - skipping destroyAllWindows")
            return
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"destroyAllWindows failed: {e}")
            self.display_enabled = False
    
    def save_image_headless(self, image, filename, directory="jetson_output"):
        """Save image when display is not available (Jetson-friendly)"""
        try:
            output_dir = Path(directory)
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / filename
            success = cv2.imwrite(str(output_path), image)
            
            if success:
                logger.info(f"Image saved to {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to save image to {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    def create_video_writer_headless(self, filename, fps, width, height, directory="jetson_output"):
        """Create video writer for headless operation (Jetson-friendly)"""
        try:
            output_dir = Path(directory)
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / filename
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if writer.isOpened():
                logger.info(f"Video writer created: {output_path}")
                return writer
            else:
                logger.error(f"Failed to create video writer: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating video writer: {e}")
            return None

# Global instance
jetson_display = JetsonDisplayHandler()

# Convenience functions
def safe_imshow(window_name, image):
    """Safe imshow that works on Jetson"""
    jetson_display.imshow(window_name, image)

def safe_namedWindow(window_name, flags=None):
    """Safe namedWindow that works on Jetson"""
    jetson_display.namedWindow(window_name, flags)

def safe_resizeWindow(window_name, width, height):
    """Safe resizeWindow that works on Jetson"""
    jetson_display.resizeWindow(window_name, width, height)

def safe_waitKey(delay):
    """Safe waitKey that works on Jetson"""
    return jetson_display.waitKey(delay)

def safe_destroyAllWindows():
    """Safe destroyAllWindows that works on Jetson"""
    jetson_display.destroyAllWindows() 