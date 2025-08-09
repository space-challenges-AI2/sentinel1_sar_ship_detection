# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
from models.rfa import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# YOLOv5 detection head
class Detect(nn.Module):
    # YOLOv5 detection head part, used to build object detection model
    stride = None  # Stride calculated during construction
    dynamic = False  # Force grid reconstruction
    export = False  # Export mode
    '''
    In YOLOv5 or similar object detection models, `anchors` is a list containing multiple anchor sizes. 
    These anchor sizes are usually organized in multiple sub-lists, where each sub-list contains anchor 
    sizes for a specific detection layer.
    `anchors[1]` refers to the second of these sub-lists, representing the anchor sizes for the second detection layer.

    For example, if `anchors` is defined as follows:
    
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    
    Here, `anchors` contains three sub-lists, each corresponding to a detection layer. In YOLOv5, these 
    detection layers are typically used to detect objects of different sizes. For example:
    
    - `anchors[0]` -> `[10, 13, 16, 30, 33, 23]` may be used to detect small-sized objects.
    - `anchors[1]` -> `[30, 61, 62, 45, 59, 119]` for medium-sized objects.
    - `anchors[2]` -> `[116, 90, 156, 198, 373, 326]` for large-sized objects.
    
    In the `anchors[1]` list, each pair of numbers represents the width and height of an anchor. 
    For example, `30` and `61` are a pair, indicating an anchor with width 30 and height 61.
    These dimensions are relative to the network input size and are usually derived from statistical 
    analysis of object sizes in the training dataset.
    
    '''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # Detection layer initialization method

        super().__init__()
        self.nc = nc  # Number of classes
        self.no = nc + 5  # Output number for each anchor
        self.nl = len(anchors)  # Number of detection layers
        self.na = len(anchors[0]) // 2  # Number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # Initialize grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # Initialize anchor grid
        # Initialize anchors
        '''
        self.register_buffer('anchors', ...) registers this tensor as a buffer of the model. 
        In PyTorch, buffers are tensors that you want to save and load with the model, but are not model parameters.
        Tensors registered as buffers are not treated as model parameters, so they won't be updated by the optimizer during training.
        This is important for values like anchors that don't need to be updated during training but are inherent parts of the model.
        '''
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # Shape (nl,na,2)
        # Initialize output head convolution layers
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # Output convolution layers
        #self.m = nn.ModuleList(DecoupledHead(x, nc, 1, anchors) for x in ch)

        self.inplace = inplace  # Use in-place operations (e.g., slice assignment)

    def forward(self, x):
        z = []  # Store output of each feature layer

        # Iterate through network layers
        for i in range(self.nl):
            # Process features from Neck output using convolution layers of output head
            x[i] = self.m[i](x[i])  # Apply convolution layer m[i] to feature map x[i]

            # Get feature map dimensions, bs is batch size, ny and nx are height and width of feature map
            bs, _, ny, nx = x[i].shape
            # Adjust feature map shape to accommodate anchor number and output number
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # Non-training mode, i.e., inference mode
            if not self.training:
                # If dynamic grid creation is needed or grid size doesn't match current feature map size
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # Create grid
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # If model is for segmentation task
                if isinstance(self, Segment):
                    # Split xy, wh, conf and mask
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # Calculate xy coordinates
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # Calculate width and height
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)  # Merge results

                else:  # For detection task
                    # Split xy, wh and conf
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # Calculate xy coordinates
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # Calculate width and height
                    y = torch.cat((xy, wh, conf), 4)  # Merge results

                # Add processed feature map to z list
                z.append(y.view(bs, self.na * nx * ny, self.no))

        # Return different results based on mode
        # Training mode: return original feature maps
        # Non-training mode: return different results based on whether model is exported
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    '''
    _make_grid
    This function is mainly used to generate two grids: one is a regular coordinate grid (grid), 
    and the other is an anchor grid (anchor_grid).
    The regular grid is used to represent the position of each cell on the feature map, while the 
    anchor grid is a grid scaled according to the anchor positions on the feature map, usually used 
    for calculating bounding boxes in object detection.
    '''
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        # Get current anchor device and data type
        d = self.anchors[i].device  # Get device where anchor is located (e.g., CPU or GPU)
        t = self.anchors[i].dtype  # Get anchor data type

        # Grid shape, where na is the number of anchors per grid
        shape = 1, self.na, ny, nx, 2  # Grid shape, where 2 represents coordinates (x, y) of each grid cell

        # Create grid cells based on the width nx and height ny of the downsampled feature map
        # torch.arange(n) generates a 1D tensor from 0 to n-1
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)  # Create y and x axis coordinate sequences respectively
        # Use meshgrid to create grid, considering PyTorch version compatibility
        # torch.meshgrid function takes two 1D tensors (y and x in this example) and generates two 2D tensors.
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # Create grid coordinate points
        # Add grid offset and expand to specified shape
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # Stack xv and yv to form grid and subtract 0.5 offset

        # Generate anchor grid, where anchors are scaled relative to feature map dimensions
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)  # Adjust anchor size and expand to grid shape

        # Return generated grid and anchor grid
        return grid, anchor_grid  # Return grid and anchor grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        # Define forward propagation, single-scale inference, used during training
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # Initialize output and time recording lists
        # Perform inference layer by layer on the model
        for m in self.model:
            if m.f != -1:  # If not from previous layer
                # Get input from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                # If performance analysis is enabled, record inference time for each layer
                self._profile_one_layer(m, x, dt)
            x = m(x)  # Execute current layer operation
            # Save output results
            y.append(x if m.i in self.save else None)
            if visualize:
                # If visualization is enabled, visualize features
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        # Evaluate inference time for each layer
        c = m == self.model[-1]  # Check if it's the last layer
        # Calculate FLOPs
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        # Log layer information
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        # Fuse Conv2d() and BatchNorm2d() layers of the model
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # Update convolution layer
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                # Remove batch normalization layer
                delattr(m, 'bn')
                # Update forward propagation function
                m.forward = m.forward_fuse
            if type(m) is PatchEmbed_FasterNet:
                m.proj = fuse_conv_and_bn(m.proj, m.norm)
                delattr(m, 'norm')  # remove BN
                m.forward = m.fuseforward
            if type(m) is PatchMerging_FasterNet:
                m.reduction = fuse_conv_and_bn(m.reduction, m.norm)
                delattr(m, 'norm')  # remove BN
                m.forward = m.fuseforward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        # Print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply operations like to(), cpu(), cuda(), half() to the model
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        # Weight initialization for Detect and Segment heads
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self



class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        # Initialize, input as model config file, input channels, number of classes, anchors
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # If cfg is a dict, use it directly as model config
        else:  # If it's a *.yaml file
            import yaml  # Import yaml module
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # Load model config from yaml file

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # Get input channels
        if nc and nc != self.yaml['nc']:
            # If number of classes is provided and different from yaml file, override
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # Override number of classes in yaml
        if anchors:
            # If anchors are provided, override anchors in yaml file
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # Override anchors in yaml
        # Parse and build model
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # Parse model, get save list
        self.names = [str(i) for i in range(self.yaml['nc'])]  # Default class names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides and anchors
        m = self.model[-1]  # Get the last module of the model, usually Detect
        if isinstance(m, (Detect, Segment)):
            s = 256  # Twice the minimum stride
            m.inplace = self.inplace
            # Instantiate forward method
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            # Calculate stride
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)  # Check anchor order
            m.anchors /= m.stride.view(-1, 1, 1)  # Adjust anchor size
            self.stride = m.stride
            # Initialize detection head biases
            self._initialize_biases()  # Run only once

        # Initialize weights and biases
        initialize_weights(self)
        self.info()  # Print model information
        LOGGER.info('')


    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        # Enhanced forward propagation
        img_size = x.shape[-2:]  # Get height and width of input image
        s = [1, 0.83, 0.67]  # Define scaling ratios
        f = [None, 3, None]  # Define flip operations (2-vertical flip, 3-horizontal flip)
        y = []  # Initialize output list

        for si, fi in zip(s, f):
            # Operate on each scaling and flip combination
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))  # Scale and flip image
            yi = self._forward_once(xi)[0]  # Perform forward propagation on processed image
            yi = self._descale_pred(yi, fi, si, img_size)  # Descale prediction results
            y.append(yi)  # Add to output list

        y = self._clip_augmented(y)  # Clip enhanced results
        return torch.cat(y, 1), None  # Return enhanced inference results for training

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # Model Detect Head initialization method
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None

# parse_model function, used to parse YOLOv5 model configuration dictionary and build model.
def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # Parse anchors, number of classes nc, depth and width multiples gd and gw, and activation function from dictionary
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # Redefine default activation function, e.g., Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # Print activation function

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # Number of anchors
    no = na * (nc + 5)  # Number of outputs = number of anchors * (number of classes + 5)

    layers, save, c2 = [], [], ch[-1]  # Initialize layer list, save list and output channels

    # Iterate through backbone and head parts of the model for parsing and building
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # Instantiate module
        '''
        In this code, the `eval` function is used for two main purposes:
    
        1. **Convert strings to modules**: When `m` is a string, `eval(m)` is used to convert this string 
           to the corresponding Python object or function. In this context, `m` is likely a string representing 
           module or class names (e.g., "Conv", "Bottleneck", etc.), and `eval(m)` converts these strings to 
           actual Python classes or functions. This is a method to dynamically create corresponding objects 
           based on string content.

        2. **Parse string expressions in parameter lists**: For each element in the `args` list, if it's a string, 
           `eval(a)` will try to evaluate the value of this string expression. This is very useful when using 
           strings to represent expressions or variable values in configuration files. For example, if an element 
           in `args` is "2 * 16", `eval` will calculate the result of this expression, which is 32.

        Overall, the `eval` function here is used to dynamically interpret and execute code fragments represented 
        by strings. This makes the code flexible to build models based on text configurations (such as YAML files).
        '''
        m = eval(m) if isinstance(m, str) else m  # Convert string to module
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # Convert string to corresponding value

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # Apply depth multiple
        # Parse module
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
                C3_CA,
                iRMB,
                RFAConv, RFCAConv, RFCBAMConv,
                BasicStage, PatchEmbed_FasterNet, PatchMerging_FasterNet,
                Conv_BN_HSwish, MobileNetV3_InvertedResidual,
                Shuffle_Block, CBRM, G_bneck,
                stem, MBConvBlock
        }:
            c1, c2 = ch[f], args[0]  # Input and output channel numbers
            if c2 != no:  # If not output layer
                '''
                The make_divisible function is typically used to ensure that a certain value can be 
                divided by another number (8 in this case). This is very useful in deep learning, 
                especially when building convolutional neural networks, because certain hardware or 
                software frameworks have specific divisibility requirements for layer input and output 
                channel numbers to ensure computational efficiency.
                '''
                c2 = make_divisible(c2 * gw, 8)  # Apply width multiple

            args = [c1, c2, *args[1:]]  # Update parameters
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, C3_CA}:
                args.insert(2, n)  # Insert repetition count
                n = 1
            elif m in [BasicStage]:
                args.pop(1)
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # Calculate concatenated channel number
        # Add bifpn_add structure
        elif m in [BiFPN_Add2, BiFPN_Add3]:
            c2 = max([ch[x] for x in f])
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])  # Add input channel numbers
            if isinstance(args[1], int):  # Number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)  # Apply width multiple
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # Create module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # Construct module
        t = str(m)[8:-2].replace('__main__.', '')  # Get module type
        np = sum(x.numel() for x in m_.parameters())  # Calculate parameter count
        # Attach index, 'from' index, type, parameter count
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # Print module information
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # Add to save list
        layers.append(m_)  # Add to layer list
        if i == 0:
            ch = []
        ch.append(c2)  # Update channel list

    return nn.Sequential(*layers), sorted(save)

class DecoupledHead(nn.Module):
	# Code is referenced from "啥都会一点的老程大佬" (Old Cheng who knows a bit of everything) https://blog.csdn.net/weixin_44119362
    def __init__(self, ch=256, nc=80, width=1.0, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.merge = Conv(ch, 256 * width, 1, 1)
        self.cls_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs1 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.reg_convs2 = Conv(256 * width, 256 * width, 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 * width, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 * width, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 * width, 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        # Classification = 3x3conv + 3x3conv + 1x1convpred
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        # Regression = 3x3conv (shared) + 3x3conv (shared) + 1x1pred
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        # Confidence = 3x3conv (shared) + 3x3conv (shared) + 1x1pred
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
