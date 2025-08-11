# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.denoising.integration import prepare_denoise_params_from_args, log_denoising_config


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or Triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0 (webcam)
        data=ROOT / 'data/coco128.yaml',  # path to dataset.yaml
        imgsz=(640, 640),  # inference image size (height, width)
        conf_thres=0.50,  # confidence threshold # TODO prev 0.25
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # maximum detections per image
        device='',  # CUDA device, e.g. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=10,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        denoise=0.0,
        denoise_method='fabf',
        denoise_rho=5.0,
        denoise_N=5,
        denoise_sigma=0.1,
        denoise_theta=None,
        denoise_clip=True
):
    # Process input source
    source = str(source)
    # Determine whether to save inference images
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # Check if input is a file
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # Check if input is a URL
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Check if input is webcam
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # Check if input is a screenshot
    screenshot = source.lower().startswith('screen')
    # If URL and file, download it
    if is_url and is_file:
        source = check_file(source)  # download

    # Handle save directory
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # Create save directory
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make directory


        # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt

    # Check image size (adjust to match model stride)
    imgsz = check_img_size(imgsz, s=stride)

    # Prepare denoising parameters
    denoise_params = {
        'enabled': denoise > 0.0,
        'probability': denoise,
        'method': denoise_method,
        'rho': denoise_rho,
        'N': denoise_N,
        'sigma': denoise_sigma,
        'theta': denoise_theta,
        'clip': denoise_clip
    }

    log_denoising_config(denoise_params)

    # Data loader
    bs = 1  # batch size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, denoise_params=denoise_params)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt, denoise_params=denoise_params)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, denoise_params=denoise_params)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup model with dummy input
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # Process each item in the dataset
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Convert image from NumPy array to PyTorch tensor and move to the selected device (CPU/GPU)
            im = torch.from_numpy(im).to(model.device)
            # Convert to float16 if enabled, otherwise to float32
            im = im.half() if model.fp16 else im.float()  # from uint8 to float16/32
            # Normalize image from 0-255 to 0.0-1.0
            im /= 255
            # If image has shape [H, W, C], add batch dimension to make it [1, C, H, W]
            if len(im.shape) == 3:
                im = im[None]

        # Inference
        with dt[1]:
            # If feature visualization is enabled, create a unique directory for each image
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # Run the model forward pass
            pred = model(im, augment=augment, visualize=visualize)

        # Non-Maximum Suppression (NMS)
        with dt[2]:
            # Apply NMS to remove redundant overlapping boxes
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # process detections for each image
            seen += 1
            if webcam:  # if using webcam, handle multiple streams
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:  # for image, video, or screenshot input
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # Set up save paths
            p = Path(p)  # convert to Path object
            save_path = str(save_dir / p.name)  # output image path
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # output label txt path
            s += '%gx%g ' % im.shape[2:]  # log image size
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain: [width, height, width, height]
            imc = im0.copy() if save_crop else im0  # copy image for cropping if enabled

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):

                num_detections = len(det)
                print(f"[INFO] Detected {num_detections} object(s) in {p.name}")

                # Rescale bounding boxes from inference size to original image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print detection summary
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # number of detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # append to status string
                  
                # Write each detection result
                for *xyxy, conf, cls in reversed(det):


                    class_id = int(cls)
                    class_name = names[class_id]
                    print(f" - Class: {class_name} (ID: {class_id}), Confidence: {conf:.3f}")

                    # Display the pixel coordinate of the ship:
                    x1, y1, x2, y2 = [x.item() for x in xyxy]
                    # centre point computation:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    print(f" - Class: {names[int(cls)]}, Conf: {conf:.3f}, Center Pixel: ({cx}, {cy})")

                    # Save detection results to .txt file
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # convert to normalized xywh format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # format: class x y w h [conf]
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Draw bounding box on image
                    if save_img or save_crop or view_img:
                        c = int(cls)  # convert class to int
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.box_label(xyxy, label, color=(0, 0, 0))  # alternative: fixed black color

                    # Save cropped object if enabled
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


            # Display results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow resizable window in Linux
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # wait 1 ms

            # Save detection result (image with bounding boxes)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # for video or stream input
                    if vid_path[i] != save_path:  # new video file
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video file
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream input
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # Set video save path and format
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force output video to use .mp4 extension
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Log inference time for this image/frame
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Log overall performance stats
    t = tuple(x.t / seen * 1E3 for x in dt)  # average time per image in ms
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    # Log where results were saved
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    # Optionally strip optimizer state from model to reduce size
    if update:
        strip_optimizer(weights[0])  # remove optimizer state to suppress SourceChangeWarning


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/YOLOv5s/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/best.pt',help='model path or triton URL')
    parser.add_argument('--source', type=str, default='.\\source', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/HRSID_land.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

    # Denoising arguments
    parser.add_argument('--denoise', type=float, default=0.0, help='probability of applying denoising (0.0 to 1.0)')
    parser.add_argument('--denoise-method', type=str, default='fabf', help='denoising method: fabf or custom')
    parser.add_argument('--denoise-rho', type=float, default=5.0, help='FABF spatial window radius')
    parser.add_argument('--denoise-N', type=int, default=5, help='FABF polynomial order')
    parser.add_argument('--denoise-sigma', type=float, default=0.1, help='FABF noise level')
    parser.add_argument('--denoise-theta', type=float, default=None, help='FABF target intensity')
    parser.add_argument('--denoise-clip', action='store_true', default=True, help='FABF clip output to [0, 1]')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
