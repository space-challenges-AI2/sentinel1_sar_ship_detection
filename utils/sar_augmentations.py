# SAR-specific augmentation functions for HRSID dataset
# Designed to increase dataset size while maintaining label integrity

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional
import torch


class SARAugmentations:
    """
    SAR-specific augmentation class for HRSID dataset
    Handles both image and label transformations
    """
    
    def __init__(self, hyp: dict):
        self.hyp = hyp
        self.cutout_prob = hyp.get('cutout_prob', 0.3)
        self.noise_prob = hyp.get('noise_prob', 0.2)
        self.blur_prob = hyp.get('blur_prob', 0.15)
        self.sharpen_prob = hyp.get('sharpen_prob', 0.15)
        self.elastic_prob = hyp.get('elastic_prob', 0.1)
        
    def __call__(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SAR-specific augmentations
        
        Args:
            img: Input image (H, W, C)
            labels: Labels in format [class, x_center, y_center, width, height] (normalized)
            
        Returns:
            Augmented image and labels
        """

        img_aug = img.copy()
        labels_aug = labels.copy()
        
        # Apply SAR-specific augmentations (these don't affect labels)
        if random.random() < self.noise_prob:
            img_aug = self.add_sar_noise(img_aug)
            
        if random.random() < self.blur_prob:
            img_aug = self.apply_blur(img_aug)
            
        if random.random() < self.sharpen_prob:
            img_aug = self.apply_sharpen(img_aug)
            
        # Apply augmentations that affect labels
        if random.random() < self.cutout_prob:
            img_aug, labels_aug = self.apply_sar_realistic_cutout(img_aug, labels_aug)
            
        if random.random() < self.elastic_prob:
            img_aug, labels_aug = self.apply_elastic_deformation(img_aug, labels_aug)
            
        return img_aug, labels_aug
    
    def add_sar_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Add SAR-specific noise (speckle-like)
        """
        # Convert to float for noise addition
        img_float = img.astype(np.float32) / 255.0
        
        # Add multiplicative speckle noise (common in SAR)
        noise_factor = random.uniform(0.05, 0.15)
        speckle_noise = np.random.exponential(1.0, img_float.shape)
        img_noisy = img_float * (1 + noise_factor * (speckle_noise - 1))
        
        # Add additive Gaussian noise
        gaussian_noise = np.random.normal(0, 0.02, img_float.shape)
        img_noisy = img_noisy + gaussian_noise
        
        # Clip to valid range and convert back
        img_noisy = np.clip(img_noisy, 0, 1)
        return (img_noisy * 255).astype(np.uint8)
    
    def apply_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur (simulates atmospheric effects)
        """
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 1.5)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    def apply_sharpen(self, img: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges
        """
        # Create unsharp mask
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def apply_sar_realistic_cutout(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SAR-realistic cutout (simulates terrain shadows, interference, etc.)
        """
        h, w = img.shape[:2]
        n_cutouts = random.randint(1, 2)
        
        for _ in range(n_cutouts):
            # Smaller, more realistic cutouts
            cutout_h = random.randint(int(h * 0.05), int(h * 0.15))
            cutout_w = random.randint(int(w * 0.05), int(w * 0.15))
            
            x = random.randint(0, w - cutout_w)
            y = random.randint(0, h - cutout_h)
            
            # Simulate radar shadow (darker region) instead of random gray
            shadow_intensity = random.randint(20, 60)  # Darker than random gray
            img[y:y + cutout_h, x:x + cutout_w] = shadow_intensity
            
            # Remove labels that are heavily obscured by cutout
            if len(labels) > 0:
                labels = self._filter_obscured_labels(labels, x, y, cutout_w, cutout_h, w, h)
        
        return img, labels  # Return both image and labels
    
    def _filter_obscured_labels(self, labels: np.ndarray, cutout_x: int, cutout_y: int, 
                               cutout_w: int, cutout_h: int, img_w: int, img_h: int) -> np.ndarray:
        """
        Filter out labels that are heavily obscured by cutout
        """
        if len(labels) == 0:
            return labels
            
        # Convert normalized coordinates to pixel coordinates
        labels_pixel = labels.copy()
        labels_pixel[:, 1] = labels[:, 1] * img_w  # x_center
        labels_pixel[:, 2] = labels[:, 2] * img_h  # y_center
        labels_pixel[:, 3] = labels[:, 3] * img_w  # width
        labels_pixel[:, 4] = labels[:, 4] * img_h  # height
        
        # Calculate intersection over area (IoA)
        valid_labels = []
        for label in labels_pixel:
            cls, x_c, y_c, w, h = label
            
            # Convert center coordinates to box coordinates
            x1 = x_c - w/2
            y1 = y_c - h/2
            x2 = x_c + w/2
            y2 = y_c + h/2
            
            # Calculate intersection with cutout
            cutout_x1, cutout_y1 = cutout_x, cutout_y
            cutout_x2, cutout_y2 = cutout_x + cutout_w, cutout_y + cutout_h
            
            # Intersection
            x_left = max(x1, cutout_x1)
            y_top = max(y1, cutout_y1)
            x_right = min(x2, cutout_x2)
            y_bottom = min(y2, cutout_y2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                label_area = w * h
                ioa = intersection / label_area
                
                # Keep label if less than 60% obscured
                if ioa < 0.6:
                    valid_labels.append(label)
            else:
                # No intersection, keep label
                valid_labels.append(label)
        
        if len(valid_labels) > 0:
            # Convert back to normalized coordinates
            valid_labels = np.array(valid_labels)
            valid_labels[:, 1] /= img_w
            valid_labels[:, 2] /= img_h
            valid_labels[:, 3] /= img_w
            valid_labels[:, 4] /= img_h
            return valid_labels
        else:
            return np.empty((0, 5))
    
    def apply_elastic_deformation(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply elastic deformation (simulates terrain variations)
        """
        h, w = img.shape[:2]
        
        # Create displacement fields
        displacement_x = np.random.randn(h, w) * random.uniform(5, 15)
        displacement_y = np.random.randn(h, w) * random.uniform(5, 15)
        
        # Smooth displacement fields
        displacement_x = cv2.GaussianBlur(displacement_x, (15, 15), 0)
        displacement_y = cv2.GaussianBlur(displacement_y, (15, 15), 0)
        
        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Apply displacement
        x_coords = np.clip(x_coords + displacement_x, 0, w - 1).astype(np.float32)
        y_coords = np.clip(y_coords + displacement_y, 0, h - 1).astype(np.float32)
        
        # Remap image
        img_deformed = cv2.remap(img, x_coords, y_coords, cv2.INTER_LINEAR)
        
        # Transform labels accordingly
        if len(labels) > 0:
            labels_deformed = self._transform_labels_elastic(labels, displacement_x, displacement_y, h, w)
        else:
            labels_deformed = labels
            
        return img_deformed, labels_deformed
    
    def _transform_labels_elastic(self, labels: np.ndarray, disp_x: np.ndarray, disp_y: np.ndarray, 
                                 h: int, w: int) -> np.ndarray:
        """
        Transform label coordinates according to elastic deformation
        """
        labels_transformed = labels.copy()
        
        for i, label in enumerate(labels_transformed):
            cls, x_c, y_c, width, height = label
            
            # Convert to pixel coordinates
            x_pixel = int(x_c * w)
            y_pixel = int(y_c * h)
            
            # Get displacement at label center
            if 0 <= y_pixel < h and 0 <= x_pixel < w:
                dx = disp_x[y_pixel, x_pixel]
                dy = disp_y[y_pixel, x_pixel]
                
                # Apply displacement to center
                new_x = (x_pixel + dx) / w
                new_y = (y_pixel + dy) / h
                
                # Clip to valid range
                new_x = np.clip(new_x, 0, 1)
                new_y = np.clip(new_y, 0, 1)
                
                labels_transformed[i, 1] = new_x
                labels_transformed[i, 2] = new_y
        
        return labels_transformed


def create_sar_augmentation_pipeline(hyp: dict) -> SARAugmentations:
    """
    Create SAR augmentation pipeline from hyperparameters
    
    Args:
        hyp: Hyperparameters dictionary
        
    Returns:
        SARAugmentations instance
    """
    return SARAugmentations(hyp)


def apply_sar_augmentations(img: np.ndarray, labels: np.ndarray, hyp: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply SAR augmentations
    
    Args:
        img: Input image
        labels: Input labels
        hyp: Hyperparameters dictionary
        
    Returns:
        Augmented image and labels
    """

    aug_pipeline = create_sar_augmentation_pipeline(hyp)
    return aug_pipeline(img, labels)


# Additional utility functions for dataset expansion
def create_synthetic_samples(img: np.ndarray, labels: np.ndarray, n_samples: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create synthetic samples through various transformations
    """
    samples = []
    
    for i in range(n_samples):
        # Simple transformations that are easier to handle
        if i == 0:
            # Horizontal flip
            img_transformed = cv2.flip(img, 1)  # 1 = horizontal flip
            labels_transformed = _flip_labels_horizontal(labels, img.shape[1])
        elif i == 1:
            # Vertical flip
            img_transformed = cv2.flip(img, 0)  # 0 = vertical flip
            labels_transformed = _flip_labels_vertical(labels, img.shape[0])
        else:
            # Small rotation (keep it simple)
            angle = random.uniform(-10, 10)  # Very small rotation
            img_transformed, labels_transformed = _rotate_image_and_labels_simple(img, labels, angle)
        
        # Only add if we have valid labels
        if len(labels_transformed) > 0:
            samples.append((img_transformed, labels_transformed))
    
    return samples

def _flip_labels_horizontal(labels: np.ndarray, img_width: int) -> np.ndarray:
    """
    Flip labels horizontally (mirror image)
    """
    if len(labels) == 0:
        return labels
    
    labels_flipped = labels.copy()
    # Flip x-coordinate: x_new = 1.0 - x_old
    labels_flipped[:, 1] = 1.0 - labels[:, 1]
    return labels_flipped

def _flip_labels_vertical(labels: np.ndarray, img_height: int) -> np.ndarray:
    """
    Flip labels vertically
    """
    if len(labels) == 0:
        return labels
    
    labels_flipped = labels.copy()
    # Flip y-coordinate: y_new = 1.0 - y_old
    labels_flipped[:, 2] = 1.0 - labels[:, 2]
    return labels_flipped

def _rotate_image_and_labels_simple(img: np.ndarray, labels: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple rotation with proper label handling
    """
    if len(labels) == 0:
        return img, labels
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    img_rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    
    # Transform labels using OpenCV's perspective transform
    labels_rotated = _transform_labels_with_matrix(labels, rotation_matrix, w, h)
    
    return img_rotated, labels_rotated

def _transform_labels_with_matrix(labels: np.ndarray, transform_matrix: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Transform labels using OpenCV transformation matrix
    """
    if len(labels) == 0:
        return labels
    
    labels_transformed = []
    
    for label in labels:
        cls, x_center, y_center, width, height = label
        
        # Convert to pixel coordinates
        x_px = x_center * img_w
        y_px = y_center * img_h
        
        # Create 4 corner points of the bounding box
        w_px = width * img_w
        h_px = height * img_h
        
        # Box corners (clockwise from top-left)
        corners = np.array([
            [x_px - w_px/2, y_px - h_px/2],  # top-left
            [x_px + w_px/2, y_px - h_px/2],  # top-right
            [x_px + w_px/2, y_px + h_px/2],  # bottom-right
            [x_px - w_px/2, y_px + h_px/2]   # bottom-left
        ], dtype=np.float32)
        
        # Transform corners
        corners_transformed = cv2.transform(corners.reshape(1, -1, 2), transform_matrix).reshape(-1, 2)
        
        # Calculate new bounding box from transformed corners
        x_min, y_min = np.min(corners_transformed, axis=0)
        x_max, y_max = np.max(corners_transformed, axis=0)
        
        # New center and dimensions
        x_new = (x_min + x_max) / 2
        y_new = (y_min + y_max) / 2
        w_new = x_max - x_min
        h_new = y_max - y_min
        
        # Convert back to normalized coordinates
        x_norm = x_new / img_w
        y_norm = y_new / img_h
        w_norm = w_new / img_w
        h_norm = h_new / img_h
        
        # Check if the transformed box is still within image bounds
        if (0.05 <= x_norm <= 0.95 and 0.05 <= y_norm <= 0.95 and 
            0.01 <= w_norm <= 0.98 and 0.01 <= h_norm <= 0.98):
            labels_transformed.append([cls, x_norm, y_norm, w_norm, h_norm])
    
    return np.array(labels_transformed) if labels_transformed else np.empty((0, 5))


def apply_sar_augmentations_only(img: np.ndarray, labels: np.ndarray, hyp: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ONLY SAR-specific augmentations that don't affect labels
    Let YOLOv5 handle all geometric transformations
    """
    # These don't affect labels, so they're safe to add
    if random.random() < hyp.get('noise_prob', 0.2):
        img = add_sar_noise(img)
        
    if random.random() < hyp.get('blur_prob', 0.15):
        img = apply_blur(img)
        
    if random.random() < hyp.get('sharpen_prob', 0.15):
        img = apply_sharpen(img)
    
    return img, labels  # Labels unchanged 