"""
Sliced Inference Module for YOLOv5
Description: Implements tiled inference to improve detection accuracy on large images
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import logging
from utils.general import non_max_suppression, scale_boxes

logger = logging.getLogger(__name__)


class SlicedInference:
    """
    Sliced inference implementation for improved detection accuracy on large images.
    
    This class handles:
    1. Image tiling with configurable overlap
    2. Inference on individual tiles
    3. Result merging and NMS across tiles
    4. Coordinate transformation between tile and original image
    """
    
    def __init__(self, 
                 tile_size: int = 640,
                 overlap: float = 0.2,
                 min_tile_size: int = 320,
                 max_tile_size: int = 1024,
                 enable_adaptive_tiling: bool = True,
                 merge_strategy: str = 'nms'):
        """
        Initialize sliced inference parameters.
        
        Args:
            tile_size: Base tile size (will be adjusted based on model stride)
            overlap: Overlap ratio between tiles (0.0 to 0.5)
            min_tile_size: Minimum tile size allowed
            max_tile_size: Maximum tile size allowed
            enable_adaptive_tiling: Whether to adapt tile size based on image dimensions
            merge_strategy: Strategy for merging results ('nms', 'weighted', 'confidence')
        """
        self.tile_size = tile_size
        self.overlap = max(0.0, min(0.5, overlap))
        self.min_tile_size = min_tile_size
        self.max_tile_size = max_tile_size
        self.enable_adaptive_tiling = enable_adaptive_tiling
        self.merge_strategy = merge_strategy
        
        # Validate parameters
        if self.overlap < 0.0 or self.overlap > 0.5:
            raise ValueError("Overlap must be between 0.0 and 0.5")
        
        logger.info(f"Sliced inference initialized: tile_size={tile_size}, overlap={overlap:.2f}")
    
    def calculate_optimal_tile_size(self, image_shape: Tuple[int, int], model_stride: int) -> int:
        """
        Calculate optimal tile size based on image dimensions and model stride.
        
        Args:
            image_shape: (height, width) of input image
            model_stride: Model's stride requirement
            
        Returns:
            Optimal tile size that respects model stride and image dimensions
        """
        h, w = image_shape
        
        # Start with base tile size
        optimal_size = self.tile_size
        
        # Adjust for model stride
        optimal_size = (optimal_size // model_stride) * model_stride
        
        # Adaptive tiling based on image size
        if self.enable_adaptive_tiling:
            # For very large images, use larger tiles
            if max(h, w) > 2000:
                optimal_size = min(self.max_tile_size, optimal_size * 1.5)
            # For small images, use smaller tiles
            elif max(h, w) < 800:
                optimal_size = max(self.min_tile_size, optimal_size * 0.75)
        
        # Ensure tile size is within bounds
        optimal_size = max(self.min_tile_size, min(self.max_tile_size, optimal_size))
        
        # Final stride adjustment
        optimal_size = (int(optimal_size) // model_stride) * model_stride
        
        logger.debug(f"Optimal tile size: {optimal_size} (original: {self.tile_size})")
        return optimal_size
    
    def create_tiles(self, 
                    image: np.ndarray, 
                    tile_size: int,
                    overlap_pixels: int) -> List[Dict[str, Any]]:
        """
        Create tiles from input image with specified overlap.
        
        Args:
            image: Input image (H, W, C)
            tile_size: Size of each tile
            overlap_pixels: Overlap in pixels between tiles
            
        Returns:
            List of tile dictionaries with coordinates and data
        """
        h, w = image.shape[:2]
        tiles = []
        
        # Calculate step size (non-overlapping portion)
        step_size = tile_size - overlap_pixels
        
        # Ensure we cover the entire image
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                # Calculate tile boundaries
                y1 = y
                x1 = x
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                
                # Extract tile
                tile = image[y1:y2, x1:x2]
                
                # Skip if tile is too small
                if tile.shape[0] < tile_size // 2 or tile.shape[1] < tile_size // 2:
                    continue
                
                # Pad tile if necessary to maintain consistent size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded_tile = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append({
                    'data': tile,
                    'coords': (x1, y1, x2, y2),
                    'original_shape': (y2 - y1, x2 - x1),
                    'padded': tile.shape != (y2 - y1, x2 - x1)
                })
        
        logger.info(f"Created {len(tiles)} tiles from image {w}x{h}")
        return tiles
    
    def transform_predictions(self, 
                        predictions: torch.Tensor,
                        tile_coords: Tuple[int, int, int, int],
                        original_shape: Tuple[int, int],
                        tile_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Transform predictions from tile coordinates to original image coordinates.
        
        Args:
            predictions: Model predictions for tile (N, 6) where 6 = [x1, y1, x2, y2, conf, class]
            tile_coords: (x1, y1, x2, y2) of tile in original image
            original_shape: (height, width) of original tile before padding
            tile_shape: (height, width) of padded tile
            
        Returns:
            Transformed predictions in original image coordinates
        """
        if len(predictions) == 0:
            return predictions
        
        # Extract coordinates
        x1, y1, x2, y2 = tile_coords
        orig_h, orig_w = original_shape
        tile_h, tile_w = tile_shape
        
        # Create a copy to avoid modifying original
        transformed = predictions.clone()
        
        # FIXED: Scale coordinates from padded tile size (640x640) to original tile size
        # The model predicts in 640x640 space, but we need to map to original tile dimensions
        scale_x = orig_w / tile_w  # orig_w / 640
        scale_y = orig_h / tile_h  # orig_h / 640
        
        # Transform bounding box coordinates
        # First scale from padded tile (640x640) to original tile dimensions
        transformed[:, 0] = transformed[:, 0] * scale_x  # x1
        transformed[:, 1] = transformed[:, 1] * scale_y  # y1
        transformed[:, 2] = transformed[:, 2] * scale_x  # x2
        transformed[:, 3] = transformed[:, 3] * scale_y  # y2
        
        # Then offset by tile position in original image
        transformed[:, 0] += x1  # x1
        transformed[:, 1] += y1  # y1
        transformed[:, 2] += x1  # x2
        transformed[:, 3] += y1  # y2
        
        # Clip coordinates to tile boundaries (optional safety check)
        transformed[:, 0] = torch.clamp(transformed[:, 0], x1, x2)
        transformed[:, 1] = torch.clamp(transformed[:, 1], y1, y2)
        transformed[:, 2] = torch.clamp(transformed[:, 2], x1, x2)
        transformed[:, 3] = torch.clamp(transformed[:, 3], y1, y2)
        
        return transformed
    
    def merge_predictions(self, 
                         all_predictions: List[torch.Tensor],
                         conf_thres: float = 0.25,
                         iou_thres: float = 0.45,
                         max_det: int = 1000) -> torch.Tensor:
        """
        Merge predictions from all tiles using specified strategy.
        
        Args:
            all_predictions: List of prediction tensors from all tiles
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            
        Returns:
            Merged and filtered predictions
        """
        if not all_predictions:
            return torch.empty((0, 6), device=all_predictions[0].device if all_predictions else 'cpu')
        
        # Filter by confidence first
        filtered_predictions = []
        for pred in all_predictions:
            if len(pred) > 0:
                mask = pred[:, 4] >= conf_thres
                filtered_predictions.append(pred[mask])
        
        if not filtered_predictions:
            return torch.empty((0, 6), device=all_predictions[0].device)
        
        # Concatenate all predictions
        merged = torch.cat(filtered_predictions, dim=0)
        
        if len(merged) == 0:
            return merged
        
        # Apply NMS
        if self.merge_strategy == 'nms':
            # Use YOLOv5's non_max_suppression
            try:
                from utils.general import non_max_suppression
                # Ensure proper format for non_max_suppression
                if len(merged.shape) == 2 and merged.shape[1] == 6:
                    # Add batch dimension and ensure proper format
                    merged_batch = merged.unsqueeze(0)  # Add batch dimension
                    merged = non_max_suppression(
                        merged_batch,
                        conf_thres=conf_thres,
                        iou_thres=iou_thres,
                        max_det=max_det
                    )[0]  # Remove batch dimension
                else:
                    logger.warning(f"Unexpected prediction format: {merged.shape}, using custom NMS")
                    merged = self._custom_nms(merged, iou_thres)
            except Exception as e:
                logger.warning(f"Failed to use YOLOv5 NMS: {e}, using custom NMS")
                merged = self._custom_nms(merged, iou_thres)
        
        elif self.merge_strategy == 'weighted':
            # Weighted average of overlapping detections
            merged = self._weighted_merge(merged, iou_thres)
        
        elif self.merge_strategy == 'confidence':
            # Keep highest confidence detection for overlapping boxes
            merged = self._confidence_based_merge(merged, iou_thres)
        
        # Limit to max_det
        if len(merged) > max_det:
            # Sort by confidence and keep top detections
            _, indices = torch.sort(merged[:, 4], descending=True)
            merged = merged[indices[:max_det]]
        
        logger.info(f"Merged {len(merged)} detections from {len(all_predictions)} tiles")
        return merged
    
    def _weighted_merge(self, predictions: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """
        Merge overlapping detections using weighted average.
        
        Args:
            predictions: Predictions tensor
            iou_thres: IoU threshold for considering overlap
            
        Returns:
            Merged predictions
        """
        if len(predictions) <= 1:
            return predictions
        
        # Calculate IoU matrix
        boxes = predictions[:, :4]
        iou_matrix = self._calculate_iou_matrix(boxes)
        
        # Find overlapping groups
        merged = []
        used = set()
        
        for i in range(len(predictions)):
            if i in used:
                continue
            
            # Find overlapping detections
            overlap_group = [i]
            used.add(i)
            
            for j in range(i + 1, len(predictions)):
                if j not in used and iou_matrix[i, j] > iou_thres:
                    overlap_group.append(j)
                    used.add(j)
            
            if len(overlap_group) == 1:
                merged.append(predictions[i])
            else:
                # Weighted average of overlapping detections
                group_preds = predictions[overlap_group]
                weights = group_preds[:, 4]  # Use confidence as weights
                
                # Weighted average of boxes
                weighted_boxes = (group_preds[:, :4] * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
                
                # Max confidence and most common class
                max_conf = weights.max()
                class_counts = torch.bincount(group_preds[:, 5].long())
                most_common_class = torch.argmax(class_counts)
                
                merged_pred = torch.cat([
                    weighted_boxes,
                    max_conf.unsqueeze(0),
                    most_common_class.float().unsqueeze(0)
                ])
                merged.append(merged_pred)
        
        return torch.stack(merged) if merged else torch.empty((0, 6), device=predictions.device)
    
    def _confidence_based_merge(self, predictions: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """
        Merge overlapping detections by keeping the highest confidence one.
        
        Args:
            predictions: Predictions tensor
            iou_thres: IoU threshold for considering overlap
            
        Returns:
            Merged predictions
        """
        if len(predictions) <= 1:
            return predictions
        
        # Sort by confidence (descending)
        _, indices = torch.sort(predictions[:, 4], descending=True)
        sorted_predictions = predictions[indices]
        
        # Calculate IoU matrix
        boxes = sorted_predictions[:, :4]
        iou_matrix = self._calculate_iou_matrix(boxes)
        
        # Keep non-overlapping detections
        kept = []
        used = set()
        
        for i in range(len(sorted_predictions)):
            if i in used:
                continue
            
            kept.append(sorted_predictions[i])
            used.add(i)
            
            # Mark overlapping detections as used
            for j in range(i + 1, len(sorted_predictions)):
                if j not in used and iou_matrix[i, j] > iou_thres:
                    used.add(j)
        
        return torch.stack(kept) if kept else torch.empty((0, 6), device=predictions.device)
    
    def _calculate_iou_matrix(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU matrix between all pairs of boxes.
        
        Args:
            boxes: Boxes tensor (N, 4) in xyxy format
            
        Returns:
            IoU matrix (N, N)
        """
        n = len(boxes)
        iou_matrix = torch.zeros((n, n), device=boxes.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._calculate_iou(boxes[i], boxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        return iou_matrix
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Calculate IoU between two boxes.
        
        Args:
            box1: First box (4,) in xyxy format
            box2: Second box (4,) in xyxy format
            
        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_image(self, 
                     image: np.ndarray,
                     model,
                     model_stride: int,
                     conf_thres: float = 0.25,
                     iou_thres: float = 0.45,
                     max_det: int = 1000,
                     device: str = 'cpu') -> torch.Tensor:
        """
        Process image using sliced inference.
        
        Args:
            image: Input image (H, W, C)
            model: YOLOv5 model
            model_stride: Model's stride requirement
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
            device: Device to run inference on
            
        Returns:
            Final predictions in original image coordinates
        """
        h, w = image.shape[:2]
        
        # Calculate optimal tile size
        tile_size = self.calculate_optimal_tile_size((h, w), model_stride)
        overlap_pixels = int(tile_size * self.overlap)
        
        logger.info(f"Processing image {w}x{h} with tiles {tile_size}x{tile_size}, overlap {overlap_pixels}px")
        
        # Create tiles
        tiles = self.create_tiles(image, tile_size, overlap_pixels)
        
        if not tiles:
            logger.warning("No tiles created, returning empty predictions")
            return torch.empty((0, 6), device=device)
        
        # Process each tile
        all_predictions = []
        
        for i, tile_info in enumerate(tiles):
            tile = tile_info['data']
            coords = tile_info['coords']
            original_shape = tile_info['original_shape']
            
            logger.debug(f"Processing tile {i+1}/{len(tiles)} at {coords}")
            
            # Prepare tile for model input
            tile_tensor = self._prepare_tile_for_inference(tile, device)
            
            # Run inference
            with torch.no_grad():
                model_output = model(tile_tensor)
                
                # Handle different model output formats
                if isinstance(model_output, (list, tuple)):
                    predictions = model_output[0]  # First element
                else:
                    predictions = model_output
                
                # Ensure predictions have the right shape
                if len(predictions.shape) == 3:
                    # Shape: [batch, num_predictions, 6]
                    predictions = predictions[0]  # Remove batch dimension
                elif len(predictions.shape) == 2:
                    # Shape: [num_predictions, 6] - already correct
                    pass
                else:
                    logger.warning(f"Unexpected prediction shape: {predictions.shape}")
                    continue
                
                # Apply confidence threshold to raw predictions
                if len(predictions) > 0:
                    # Filter by confidence (assuming confidence is in column 4)
                    conf_mask = predictions[:, 4] >= conf_thres
                    predictions = predictions[conf_mask]
                

                
                # Transform predictions to original image coordinates
                if len(predictions) > 0:
                    transformed_predictions = self.transform_predictions(
                        predictions, coords, original_shape, tile.shape[:2]
                    )
                    all_predictions.append(transformed_predictions)
        
        # Merge predictions from all tiles
        final_predictions = self.merge_predictions(
            all_predictions, conf_thres, iou_thres, max_det
        )
        
        return final_predictions
    
    def _prepare_tile_for_inference(self, tile: np.ndarray, device: str) -> torch.Tensor:
        """
        Prepare tile for model inference.
        
        Args:
            tile: Input tile (H, W, C)
            device: Target device
            
        Returns:
            Prepared tensor for model input
        """
        # Convert to tensor and move to device
        tile_tensor = torch.from_numpy(tile).to(device)
        
        # Convert to float and normalize
        tile_tensor = tile_tensor.float() / 255.0
        
        # Add batch dimension and reorder to (B, C, H, W)
        if len(tile_tensor.shape) == 3:
            tile_tensor = tile_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tile_tensor

    def _custom_nms(self, predictions: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """
        Custom NMS implementation as fallback when YOLOv5 NMS fails.
        
        Args:
            predictions: Predictions tensor (N, 6)
            iou_thres: IoU threshold
            
        Returns:
            Filtered predictions
        """
        if len(predictions) <= 1:
            return predictions
        
        # Sort by confidence (descending)
        _, indices = torch.sort(predictions[:, 4], descending=True)
        sorted_predictions = predictions[indices]
        
        # Calculate IoU matrix
        boxes = sorted_predictions[:, :4]
        iou_matrix = self._calculate_iou_matrix(boxes)
        
        # Keep non-overlapping detections
        kept = []
        used = set()
        
        for i in range(len(sorted_predictions)):
            if i in used:
                continue
            
            kept.append(sorted_predictions[i])
            used.add(i)
            
            # Mark overlapping detections as used
            for j in range(i + 1, len(sorted_predictions)):
                if j not in used and iou_matrix[i, j] > iou_thres:
                    used.add(j)
        
        return torch.stack(kept) if kept else torch.empty((0, 6), device=predictions.device)


def create_sliced_inference_from_args(args) -> Optional[SlicedInference]:
    """
    Create SlicedInference instance from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        SlicedInference instance or None if disabled
    """
    if not getattr(args, 'sliced_inference', False):
        return None
    
    return SlicedInference(
        tile_size=getattr(args, 'sliced_tile_size', 640),
        overlap=getattr(args, 'sliced_overlap', 0.2),
        min_tile_size=getattr(args, 'sliced_min_tile_size', 320),
        max_tile_size=getattr(args, 'sliced_max_tile_size', 1024),
        enable_adaptive_tiling=getattr(args, 'sliced_adaptive', True),
        merge_strategy=getattr(args, 'sliced_merge_strategy', 'nms')
    )


def log_sliced_inference_config(sliced_inference: SlicedInference, logger=None):
    """
    Log sliced inference configuration.
    
    Args:
        sliced_inference: SlicedInference instance
        logger: Logger instance (optional)
    """
    if sliced_inference is None:
        return
    
    config_str = f"Sliced Inference: tile_size={sliced_inference.tile_size}, "
    config_str += f"overlap={sliced_inference.overlap:.2f}, "
    config_str += f"merge_strategy={sliced_inference.merge_strategy}, "
    config_str += f"adaptive={sliced_inference.enable_adaptive_tiling}"
    
    if logger:
        logger.info(config_str)
    else:
        print(config_str) 