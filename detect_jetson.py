#!/usr/bin/env python3
"""
Jetson-optimized detection script for SAR Ship Detection
Handles ARM64 compatibility and provides Jetson-specific optimizations
Author: @amanarora9848
"""

import argparse
import os
import sys
import platform
from pathlib import Path
import time
import torch
import numpy as np
import cv2

# Add the project root to the path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes,
                          check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.jetson_display import safe_imshow, safe_namedWindow, safe_resizeWindow, safe_waitKey, safe_destroyAllWindows
from utils.jetson_math import safe_random_uniform, safe_interpolation

def check_jetson_environment():
    """Check if running in Jetson environment"""
    try:
        import platform
        is_arm64 = platform.machine() == 'aarch64'
        
        # Check for Jetson-specific files
        jetson_files = ['/etc/nv_tegra_release', '/sys/module/tegra_fuse/parameters/tegra_chip_id']
        is_jetson = any(Path(f).exists() for f in jetson_files)
        
        if is_arm64 and is_jetson:
            LOGGER.info(f"{colorstr('Jetson detected:')} ARM64 architecture with Jetson hardware")
            return True
        elif is_arm64:
            LOGGER.info(f"{colorstr('ARM64 detected:')} Running on ARM64 architecture")
            return True
        else:
            LOGGER.info(f"{colorstr('x86_64 detected:')} Running on x86_64 architecture")
            return False
            
    except Exception as e:
        LOGGER.warning(f"Could not detect architecture: {e}")
        return False

def optimize_for_jetson(model, device):
    """Apply Jetson-specific optimizations"""
    try:
        if device.type == 'cuda':
            # Enable mixed precision for Jetson
            if hasattr(torch, 'autocast'):
                LOGGER.info("Enabling mixed precision (FP16) for Jetson")
            
            # Set optimal CUDA settings for Jetson
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Use FP16 if available
            if hasattr(model, 'half'):
                model.half()
                LOGGER.info("Model converted to FP16 for Jetson optimization")
            
            LOGGER.info("Jetson optimizations applied")
        else:
            LOGGER.info("Running on CPU - Jetson optimizations not applicable")
            
    except Exception as e:
        LOGGER.warning(f"Some Jetson optimizations failed: {e}")

def safe_inference(model, im, device, half=False):
    """Safe inference with Jetson compatibility checks"""
    try:
        # Ensure input is on correct device
        if im.device != device:
            im = im.to(device)
        
        # Check for invalid inputs
        if torch.isnan(im).any() or torch.isinf(im).any():
            LOGGER.error("Invalid input detected, aborting inference")
            return None
        
        # Run inference
        with torch.no_grad():
            if half and device.type == 'cuda':
                im = im.half()
            
            pred = model(im)
        
        # Validate output
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            LOGGER.error("Invalid output detected")
            return None
        
        return pred
        
    except Exception as e:
        LOGGER.error(f"Inference failed: {e}")
        return None

def run_detection(weights='runs/train/experiment4/weights/best.pt',
                 source='test_ingest',
                 data='data/HRSID.yaml',
                 imgsz=(640, 640),
                 conf_thres=0.25,
                 iou_thres=0.45,
                 max_det=1000,
                 device='',
                 view_img=False,
                 save_txt=False,
                 save_conf=False,
                 save_crop=False,
                 nosave=False,
                 classes=None,
                 agnostic_nms=False,
                 augment=False,
                 visualize=False,
                 update=False,
                 project='runs/detect',
                 name='exp',
                 exist_ok=False,
                 line_thickness=3,
                 hide_labels=False,
                 hide_conf=False,
                 half=False,
                 dnn=False,
                 vid_stride=1):
    
    # Check Jetson environment
    is_jetson = check_jetson_environment()
    
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Optimize for Jetson if applicable
    if is_jetson:
        optimize_for_jetson(model, device)
    
    # Data loader
    if os.path.isdir(source):
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    
    # Create output directory
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    # Process images
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        
        # Convert image
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # Inference
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Safe inference with Jetson compatibility
        pred = safe_inference(model, im, device, half)
        if pred is None:
            LOGGER.warning(f"Skipping {path} due to inference failure")
            continue
        
        # NMS
        t3 = time_sync()
        dt[1] += t3 - t2
        
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # Process detections
        for i, det in enumerate(pred):
            seen += 1
            if os.path.isdir(source):
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_crop or not nosave:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            # Stream results
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    safe_namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    safe_resizeWindow(str(p), im0.shape[1], im0.shape[0])
                safe_imshow(str(p), im0)
                safe_waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if not nosave:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or not nosave:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/experiment4/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='test_ingest', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='data/HRSID.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    
    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    
    run_detection(**vars(args))

if __name__ == "__main__":
    main() 