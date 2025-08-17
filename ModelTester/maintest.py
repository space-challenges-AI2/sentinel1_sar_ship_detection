import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

# --- Configuration & Page Setup ---
st.set_page_config(
    page_title="YOLOv5 Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Core Helper Functions ---

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [xmin, ymin, xmax, ymax] format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Handle the case of zero area to avoid division by zero
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0

    # Compute the IoU
    iou = interArea / unionArea
    return iou

def load_yolo_labels(label_path, img_w, img_h):
    """
    Loads YOLO format labels and converts them to [xmin, ymin, xmax, ymax] format.
    """
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # YOLO format: class_id x_center y_center width height
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            
            xmin = x_center - (width / 2)
            ymin = y_center - (height / 2)
            xmax = x_center + (width / 2)
            ymax = y_center + (height / 2)
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def evaluate_performance(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculates TP, FP, FN for a single image.
    """
    tp = 0
    fp = 0
    
    # If no ground truths, all predictions are false positives
    if not ground_truths:
        fp = len(predictions)
        fn = 0
        return {'tp': tp, 'fp': fp, 'fn': fn}

    used_gt = [False] * len(ground_truths)
    
    # Match predictions to ground truths
    for pred_box in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(ground_truths):
            # Only consider unused ground truths for matching
            if not used_gt[i]:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            used_gt[best_gt_idx] = True # Mark this ground truth as used
        else:
            fp += 1
    
    # Count unused ground truths as false negatives
    fn = len(ground_truths) - sum(used_gt)
    
    return {'tp': tp, 'fp': fp, 'fn': fn}

@st.cache_resource
def load_model(model_path):
    """
    Loads the YOLOv5 model from a .pt file. Caches the loaded model.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def run_analysis(model, image_files, num_images_to_test):
    """
    Analyzes YOLOv5 performance across a range of confidence thresholds.
    """
    if not image_files:
        st.warning("No image files found for the selected dataset.")
        return pd.DataFrame()

    files_to_process = image_files[:num_images_to_test]
    st.info(f"Analyzing {len(files_to_process)} images for each threshold. This may take a moment...")

    thresholds_to_test = np.arange(0.10, 0.95, 0.05)
    analysis_results = []
    original_conf = model.conf  # Save original confidence

    progress_bar = st.progress(0)
    total_steps = len(thresholds_to_test)
    
    for i, threshold in enumerate(thresholds_to_test):
        model.conf = round(threshold, 2)
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for image_path, label_path in files_to_process:
            try:
                original_image_rgb = np.array(Image.open(image_path).convert("RGB"))
                h, w, _ = original_image_rgb.shape
                
                gt_boxes = load_yolo_labels(label_path, w, h)
                
                # Run inference
                yolo_results = model(original_image_rgb)
                yolo_predictions = [pred[:4].tolist() for pred in yolo_results.xyxy[0].cpu().numpy()]
                
                # Evaluate
                performance = evaluate_performance(yolo_predictions, gt_boxes)
                total_tp += performance['tp']
                total_fp += performance['fp']
                total_fn += performance['fn']
            except Exception as e:
                st.warning(f"Could not process {os.path.basename(image_path)}: {e}")

        # Calculate metrics for this threshold
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        analysis_results.append({
            "Threshold": f"{threshold:.2f}",
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Precision": precision, "Recall": recall, "F1-Score": f1_score
        })
        
        progress_bar.progress((i + 1) / total_steps)

    model.conf = original_conf  # Restore original confidence
    return pd.DataFrame(analysis_results)

# --- Streamlit UI ---

st.title("📊 YOLOv5 Model Performance Dashboard")
st.markdown("This dashboard analyzes the performance of a YOLOv5 model by testing different confidence thresholds.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # You can choose which model to use by default
    model_path = 'best_land.pt' # or 'best_sea.pt' etc.
    st.info(f"Using model: **{model_path}**")

    st.markdown("---")
    st.header("📁 Dataset Paths")
    
    # Sea Dataset
    st.subheader("🌊 Sea Dataset")
    sea_images_path = "sea_dataset/images"
    sea_labels_path = "sea_dataset/labels"
    
    # Land Dataset
    st.subheader("🌳 Land Dataset")
    land_images_path = "HRSID_land_main/images/val"
    ## <<< FIX HERE: The label path was pointing to the images directory.
    land_labels_path = "HRSID_land_main/labels/val" # Corrected Path

    st.markdown("---")
    
    dataset_choice = st.selectbox(
        "Choose a dataset to analyze:",
        ("Sea", "Land", "Total"),
        key="dataset_choice"
    )
    
    analyze_button = st.button("🚀 Analyze Performance")

# --- Main Dashboard Area ---

if analyze_button:
    # --- Input Validation ---
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please make sure the file exists.")
    else:
        # Load model
        with st.spinner(f"Loading YOLOv5 model from '{model_path}'..."):
            model = load_model(model_path)
        
        if model:
            st.success("Model loaded successfully!")
            
            image_files = []
            
            # --- Prepare file lists based on user choice ---
            def get_file_pairs(img_dir, lbl_dir):
                pairs = []
                if not os.path.isdir(img_dir):
                    return pairs, f"Image directory not found: {img_dir}"
                if not os.path.isdir(lbl_dir):
                    return pairs, f"Label directory not found: {lbl_dir}"
                
                # Use glob to find common image types
                image_paths = glob(os.path.join(img_dir, '*.jpg')) + \
                              glob(os.path.join(img_dir, '*.png')) + \
                              glob(os.path.join(img_dir, '*.jpeg'))
                
                for img_path in sorted(image_paths):
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
                    # Only add the pair if the corresponding label file exists
                    if os.path.exists(lbl_path):
                        pairs.append((img_path, lbl_path))
                return pairs, None

            error_msg = None
            if dataset_choice == "Sea":
                image_files, error_msg = get_file_pairs(sea_images_path, sea_labels_path)
            elif dataset_choice == "Land":
                image_files, error_msg = get_file_pairs(land_images_path, land_labels_path)
            elif dataset_choice == "Total":
                sea_files, sea_error = get_file_pairs(sea_images_path, sea_labels_path)
                land_files, land_error = get_file_pairs(land_images_path, land_labels_path)
                
                if sea_error: st.warning(f"Sea dataset issue: {sea_error}")
                if land_error: st.warning(f"Land dataset issue: {land_error}")
                
                image_files = sea_files + land_files
                if not image_files:
                    error_msg = "No valid data found for Sea or Land datasets."

            if error_msg:
                st.error(error_msg)
            elif not image_files:
                st.warning("No images with corresponding labels were found in the specified directory/directories.")
            else:
                # --- Run analysis and display results ---
                df_results = run_analysis(model, image_files, num_images_to_test=100)
                
                if not df_results.empty:
                    st.header(f"📈 Analysis Results for '{dataset_choice}' Dataset")
                    
                    # Find and highlight the best F1-score
                    best_f1_row = df_results.loc[df_results['F1-Score'].astype(float).idxmax()]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best F1-Score", f"{best_f1_row['F1-Score']:.3f}")
                    col2.metric("Optimal Threshold", f"{best_f1_row['Threshold']}")
                    col3.metric("Detections at Best F1", f"{int(best_f1_row['TP']) + int(best_f1_row['FP'])}")

                    st.markdown("#### Performance Metrics vs. Confidence Threshold")
                    
                    styled_df = df_results.style.apply(
                        lambda s: ['background-color: #ffff99' if s.name == pd.to_numeric(df_results['F1-Score']).idxmax() else '' for v in s],
                        axis=1
                    ).format({
                        "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.3f}"
                    })
                    st.dataframe(styled_df, use_container_width=True)

                    # --- Create and display the plot ---
                    st.markdown("#### Performance Plot")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Convert columns to numeric for plotting
                    df_results["Threshold_num"] = pd.to_numeric(df_results["Threshold"])
                    df_results["Precision_num"] = pd.to_numeric(df_results["Precision"])
                    df_results["Recall_num"] = pd.to_numeric(df_results["Recall"])
                    df_results["F1-Score_num"] = pd.to_numeric(df_results["F1-Score"])

                    ax.plot(df_results["Threshold_num"], df_results["Precision_num"], 'b-o', label='Precision')
                    ax.plot(df_results["Threshold_num"], df_results["Recall_num"], 'g-o', label='Recall')
                    ax.plot(df_results["Threshold_num"], df_results["F1-Score_num"], 'r-s', label='F1-Score', linewidth=3, markersize=8)
                    
                    # Highlight the best point
                    best_f1_thresh_num = float(best_f1_row['Threshold'])
                    best_f1_score_num = float(best_f1_row['F1-Score'])
                    ax.axvline(x=best_f1_thresh_num, color='grey', linestyle='--', label=f'Best F1 @ {best_f1_thresh_num:.2f}')
                    ax.plot(best_f1_thresh_num, best_f1_score_num, 'y*', markersize=15, label=f'Best F1-Score: {best_f1_score_num:.3f}')

                    ax.set_title('Model Performance vs. Confidence Threshold')
                    ax.set_xlabel('Detection Confidence Threshold')
                    ax.set_ylabel('Score')
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    ax.legend()
                    ax.set_ylim(0, 1.05)
                    
                    st.pyplot(fig)
else:
    st.info("ℹ️ Ready to start. Click the 'Analyze Performance' button in the sidebar.")