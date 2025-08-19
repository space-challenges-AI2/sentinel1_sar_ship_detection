# SAR Ship Detection Augmentation System

This document explains the enhanced augmentation system designed specifically for the HRSID_land_denoised dataset to increase the effective training dataset size from 393 to approximately 2000+ images.

## Overview

The HRSID_land_denoised dataset contains only 393 training images, which is insufficient for robust deep learning model training. This augmentation system addresses this limitation by implementing SAR-specific augmentation techniques that:

1. **Increase effective dataset size** by 5-6x
2. **Maintain label integrity** during transformations
3. **Improve model generalization** for near-land scenarios
4. **Handle SAR-specific characteristics** (speckle noise, intensity variations)

## Files Created

- `data/hyp/hyp.land_denoised_augmented.yaml` - Enhanced hyperparameters
- `utils/sar_augmentations.py` - SAR-specific augmentation functions
- `demo_augmentation.py` - Demonstration and analysis script
- `train_land_denoised_augmented.py` - Training script with augmentations

## Augmentation Techniques

### 1. Geometric Augmentations
- **Rotation**: ±15° (simulates different viewing angles)
- **Scale**: ±30% (handles various ship sizes)
- **Translation**: ±20% (accounts for positioning variations)
- **Shear**: ±5° (simulates terrain effects)
- **Perspective**: Subtle (realistic viewpoint changes)

### 2. Flip Augmentations
- **Horizontal flip**: 50% probability (ships can appear from either direction)
- **Vertical flip**: 30% probability (accounts for different radar orientations)

### 3. Advanced Augmentations
- **Mosaic**: 80% probability (combines multiple images)
- **MixUp**: 20% probability (blends images and labels)
- **Copy-paste**: 10% probability (object-level augmentation)

### 4. SAR-Specific Augmentations
- **Speckle noise**: Simulates SAR imaging artifacts
- **Gaussian blur**: Accounts for atmospheric effects
- **Unsharp masking**: Enhances edge detection
- **Cutout**: Improves robustness to occlusions
- **Elastic deformation**: Simulates terrain variations

## Label Handling

All augmentations properly transform bounding box coordinates:

- **Coordinate transformation**: Labels are updated according to image transformations
- **Boundary clipping**: Labels are clipped to image boundaries
- **Occlusion handling**: Labels heavily obscured by cutouts are filtered out
- **Format preservation**: YOLO format compatibility is maintained

## Usage

### 1. Quick Demo
```bash
python demo_augmentation.py
```
This will:
- Analyze the current dataset
- Calculate expected dataset size increase
- Demonstrate augmentations on sample images
- Generate a visualization

### 2. Training with Augmentations
```bash
python train_land_denoised_augmented.py
```
This will:
- Use the enhanced hyperparameters
- Apply all augmentations automatically
- Train for 100 epochs with smaller batch size
- Save results to `runs/train_land_denoised_augmented/`

### 3. Custom Training
```bash
python train.py \
    --data data/HRSID_land_denoised.yaml \
    --hyp data/hyp/hyp.land_denoised_augmented.yaml \
    --epochs 100 \
    --batch-size 16 \
    --weights yolov5n.pt
```

## Expected Results

### Dataset Size Increase
- **Original**: 393 training images
- **Effective**: ~2,000+ training images
- **Multiplier**: 5-6x increase

### Performance Improvements
- Better generalization to different SAR conditions
- Improved robustness to speckle noise
- Enhanced detection of ships in various orientations
- Reduced overfitting on small dataset

## Hyperparameter Tuning

The augmentation parameters can be adjusted in `data/hyp/hyp.land_denoised_augmented.yaml`:

```yaml
# Increase for more aggressive augmentation
degrees: 15.0          # Rotation range
scale: 0.3             # Scale variation
translate: 0.2         # Translation range

# Adjust probabilities
cutout_prob: 0.3       # Cutout frequency
noise_prob: 0.2        # Noise addition frequency
blur_prob: 0.15        # Blur frequency
```

## Monitoring and Validation

During training, monitor:

1. **Training loss**: Should decrease steadily
2. **Validation mAP**: Should improve without overfitting
3. **Augmentation effects**: Check that labels remain valid
4. **Memory usage**: Augmentations may increase memory requirements

## Troubleshooting

### Common Issues

1. **Labels not transforming correctly**
   - Check that labels are in YOLO format [class, x_center, y_center, width, height]
   - Ensure coordinates are normalized (0-1)

2. **Memory errors**
   - Reduce batch size
   - Reduce image size
   - Disable some augmentations

3. **Poor performance**
   - Reduce augmentation intensity
   - Increase training epochs
   - Check label quality

### Debug Mode

To debug augmentations, modify `utils/sar_augmentations.py`:

```python
# Add debug prints
print(f"Original labels: {labels}")
print(f"Augmented labels: {labels_aug}")
```

## Best Practices

1. **Start conservative**: Begin with moderate augmentation parameters
2. **Monitor validation**: Ensure augmentations don't cause overfitting
3. **Label quality**: Verify that transformed labels remain accurate
4. **Iterative improvement**: Gradually increase augmentation intensity
5. **Domain knowledge**: Consider SAR-specific characteristics when tuning

## Technical Details

### Coordinate Transformations

The system handles various coordinate transformations:

- **Rotation**: Uses rotation matrices with proper center points
- **Scaling**: Maintains aspect ratios and relative positions
- **Translation**: Clips coordinates to image boundaries
- **Elastic deformation**: Applies displacement fields to both images and labels

### Memory Management

- Augmentations are applied in-place where possible
- Large transformations (elastic deformation) use efficient numpy operations
- Labels are processed in batches to minimize memory overhead

### Performance Optimization

- Vectorized operations for label transformations
- Efficient image processing using OpenCV
- Minimal memory allocations during augmentation

## Future Enhancements

Potential improvements:

1. **Adaptive augmentation**: Adjust intensity based on training progress
2. **Domain-specific augmentations**: More SAR-specific techniques
3. **Label-aware augmentation**: Intelligent label preservation
4. **Performance optimization**: GPU-accelerated augmentations
5. **Validation augmentation**: Consistent validation set augmentation

## References

- YOLOv5 augmentation pipeline
- SAR image processing techniques
- Data augmentation for small datasets
- Label-preserving transformations

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the demo script output
3. Verify hyperparameter settings
4. Check label format and quality 