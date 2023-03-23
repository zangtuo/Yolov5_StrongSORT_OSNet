import argparse
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements,
                                  cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from trackers.multi_tracker_zoo import create_tracker

track_method = 'ocsort'
coco_classes = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

class Hir_Yolo:
    """
    HirYOLO：YOLOV5检测人和物，ocsort跟踪， coco训练
    """

    def __init__(self, yolo_weights, track_method=track_method, device='0', slice_enabled=False):
        self.yolo_weights = Path(yolo_weights)
        self.device = select_device(device)
        self.model = DetectMultiBackend(yolo_weights, device=self.device, dnn=False, data=None, fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = (640, 640)
        self.imgsz = check_img_size(self.imgsz, s=stride)
        self.track_method = track_method
        self.track_config = ROOT / 'trackers' / self.track_method / 'configs' / (self.track_method + '.yaml')
        self.reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        self.slice_enabled = slice_enabled

    def coco_all(self, source, frames=0):
        # TODO：轨迹绘制
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        dataset = LoadStreams(source, img_size=self.imgsz, stride=stride, auto=pt, vid_stride=1)  # 数据加载

        tracker = create_tracker(track_method, self.track_config, self.reid_weights, self.device, False)

        track_result = []
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None], [None]
        outputs = [None]

        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

            # read image
            with dt[0]:
                if self.slice_enabled:
                    im = im0s

                im = torch.from_numpy(im).to(self.device)
                im = im.half()
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                if self.slice_enabled:
                    # TODO: SAHI
                    pred = []
                else:
                    pred = self.model(im, augment=False, visualize=False)

            # Apply NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[7, 9], agnostic=False,
                                           max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1

                curr_frames[i] = im

                if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                             im0s[i].shape).round()  # rescale boxes to im0 size

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker.update(det.cpu(), im)
            track_result.append(outputs)
            # if frame_idx > frames:
            #     return track_result
        return track_result

    @torch.no_grad()
    def interface_people_show(self, source):
        """
        扫描框展示
        """
        hide_labels = False
        hide_class = False
        hide_conf = False
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)

        tracker = create_tracker(track_method, self.track_config, self.reid_weights, self.device, False)

        track_result = []
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None], [None]
        outputs = [None]

        w, h = [0, 0]
        n_frame = 0
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

            w = im0s.shape[1]
            h = im0s.shape[0]
            n_frame += 1

            # read image
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if False else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                if self.slice_enabled:
                    # TODO: SAHI
                    pred = []
                else:
                    pred = self.model(im, augment=False, visualize=False)

            # Apply NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                           max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                annotator = Annotator(im0, line_width=2, example=str(names))
                curr_frames[i] = im0

                if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # pass detections to strongsort
                    with dt[3]:
                        outputs[i] = tracker.update(det.cpu(), im)
                    if len(outputs[i]) > 0:
                        for j, (output) in enumerate(outputs[i]):
                            bbox = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                            # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)

                im0 = annotator.result()
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()
                LOGGER.info(
                    f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
            outputs_list = outputs[0].tolist()
            track_result.append(list(outputs_list))
        return w, h, n_frame, track_result

    def intenface_data(self, source, thing: list) -> list:
        # 返回指定人和物的坐标
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        dataset = LoadStreams(source, img_size=self.imgsz, stride=stride, auto=pt, vid_stride=1)

        tracker = create_tracker(track_method, self.track_config, self.reid_weights, self.device, False)

        track_result = []
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None], [None]
        outputs = [None]

        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

            # read image
            with dt[0]:
                if self.slice_enabled:
                    im = im0s

                im = torch.from_numpy(im).to(self.device)
                im = im.half()
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                if self.slice_enabled:
                    # TODO: SAHI
                    pred = []
                else:
                    pred = self.detection_model(im, augment=False, visualize=False)

            # Apply NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[7, 9], agnostic=False,
                                           max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1

                curr_frames[i] = im

                if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                             im0s[i].shape).round()  # rescale boxes to im0 size

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker.update(det.cpu(), im)
        if outputs[5] in thing:
            # if frame_idx > frames:
            #     return track_result
            track_result.extend(outputs)
        return track_result


# def interface_

if __name__ == '__main__':
    # 模型选择
    model_dir = '.'
    model_name = 'yolov5x'

    tracker = Hir_Yolo(yolo_weights=os.path.join(model_dir, model_name) + '.pt')
    w, h, n_frame, outputs = tracker.interface_people_show(source='media_hir')
    # w, h, n_frame, outputs_2 = tracker.interface_people_show(source='media_hir\\xingren.mp4')
    # outputs = tracker.coco_all(source='media_hir\\video.avi')
    # tracker.
    outputs
