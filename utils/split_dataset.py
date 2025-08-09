from pathlib import Path
import shutil
import random

# Set original dataset path and target path
src_dataset_path = Path('data/HRSID_land_dataset')  # Original image and label storage path
dst_dataset_path = Path('data/HRSID_land_main')

# Target folder paths
train_images_dst_path = dst_dataset_path / 'images/train'
val_images_dst_path = dst_dataset_path / 'images/val'
train_labels_dst_path = dst_dataset_path / 'labels/train'
val_labels_dst_path = dst_dataset_path / 'labels/val'

# Ensure directories exist
for path in [train_images_dst_path, val_images_dst_path, train_labels_dst_path, val_labels_dst_path]:
    path.mkdir(parents=True, exist_ok=True)

# List all original images and labels - corrected paths
images = list(src_dataset_path.glob('images/*.jpg'))  # Fixed: search in images subfolder
labels = [img.parent.parent / 'labels' / img.with_suffix('.txt').name for img in images]  # Fixed: labels are in labels subfolder

# Split dataset
data = list(zip(images, labels))
random.shuffle(data)
train_data = data[:int(0.7 * len(data))]
val_data = data[int(0.7 * len(data)):]

# Copy files to target directories - add existence check
for img, label in train_data:
    if label.exists():  # Check if label file exists
        shutil.copy(img, train_images_dst_path)
        shutil.copy(label, train_labels_dst_path)
    else:
        print(f"Warning: Label file {label} not found for image {img}")

for img, label in val_data:
    if label.exists():  # Check if label file exists
        shutil.copy(img, val_images_dst_path)
        shutil.copy(label, val_labels_dst_path)
    else:
        print(f"Warning: Label file {label} not found for image {img}")

print(f"Split complete: {len(train_data)} train, {len(val_data)} validation")