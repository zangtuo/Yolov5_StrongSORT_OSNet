import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

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

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from trackers.multi_tracker_zoo import create_tracker

'''constants'''
track_method = 'ocsort'

class YoloTracker:
    def __init__(self, yolo_weights, track_method=track_method, device='', slice_enabled=False):
        self.yolo_weights = Path(yolo_weights)
        self.device = select_device(device)
        self.detection_model = DetectMultiBackend(yolo_weights, device=self.device, dnn=False, data=None, fp16=False)
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        self.imgsz = (640,640)
        self.imgsz = check_img_size(self.imgsz, s=stride)
        self.track_method = track_method
        self.track_config = ROOT / 'trackers' / self.track_method / 'configs' / (self.track_method+ '.yaml')
        self.reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        self.slice_enabled = slice_enabled


    '''监控车辆追踪'''
    def track_webcam(self, source, frames):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
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
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[7,9], agnostic=False, max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1

                curr_frames[i] = im

                if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s[i].shape).round()  # rescale boxes to im0 size

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker.update(det.cpu(), im)

            if frame_idx > frames:
                return track_result

    '''single image detection'''
    @torch.no_grad()
    def track_img(self, source):
        pass

    '''无人机车辆追踪'''
    @torch.no_grad()
    def track_vid(self, source):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)

        tracker = create_tracker(track_method, self.track_config, self.reid_weights, self.device, False)

        track_result = []
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None], [None]
        outputs = [None]

        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

            # read image
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if False else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # if self.slice_enabled:
                #     im = im0s

            # Inference
            with dt[1]:
                if self.slice_enabled:
                    # TODO: SAHI
                    pred = []
                else:
                    pred = self.detection_model(im, augment=False, visualize=False)

            # Apply NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                           max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = im0s.copy()

                curr_frames[i] = im0

                if hasattr(tracker, 'tracker') and hasattr(tracker, 'camera_update'):
                    if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                        tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # pass detections to strongsort
                    with dt[3]:
                        outputs[i] = tracker.update(det.cpu(), im)

                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

            track_result.append(outputs)


        return track_result

class YoloTrackerAnalyzer:
    def __init__(self):
        pass

    def calc_density(self, w, h, result, grid_w, grid_h):
        density = np.zeros([grid_w, grid_h])
        for r in result:
            centroid = [(r[0] + r[2])/2, (r[1] + r[3])/2]
            [x,y] = centroid
            idx_x = int(np.floor(x/grid_w))
            idx_y = int(np.floor(y/grid_h))
            density[idx_x, idx_y] += 1

        return density

    def esti_trend(self, w, h, detection):
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        val = 0
        trend = [x1, y1, x2, y2, val]
        trends = [trend]
        return trends


    def esti_agglo(self, w, h, result):
        threshold = 20
        density = np.array(self.calc_density(w, h, result, 16, 16))

        # compute mean, std, and media of density
        mu_d = np.average(density)
        std_d = np.std(density)
        median_d = np.median(density)

        # TODO if a grid's density is above the threshold && higher than 3*std_d+median_d, consider it as agglomerated

    def esti_flee(self, w, h, results):
        density = []
        for result in results:
            dense = np.array(self.calc_density(w, h, result, 16, 16))
            density.append(dense)

        density = np.array(density)

        #TODO calculate

    def esti_conflict(self, w, h, result):
        pass



class YoloTrackerVisualizer:
    def draw_track(self, w, h, tracks):
        # tracks smoothing
        pass

    def draw_boundbox(self, w, h, result):
        pass

    def draw_vid(self, w, h, video, result, file_save, draw_bb=True, draw_track=False, track_frame=10):
        pass


if __name__ == '__main__':
    tracker = YoloTracker(yolo_weights='.\\visdrone.pt')

    w, h, n_frame, outputs = tracker.track_vid('media\\vid3.mp4')

    '''density calculation'''
    density = YoloTrackerAnalyzer.calc_density(w, h, outputs[-1,:,:], 10, 10)

    '''track draw based on the last frame of video'''
    # tracks = YoloTrackerAnalyzer.calc_last_track(w, h, outputs, frame_recur)
    # track_img = YoloTrackerAnalyzer.draw_last_track(w, h, outputs)

    # true_north = 100
    # YoloTrackerAnalyzer.calc_trend(w, h, outputs, true_north)

    # w, h, outputs = tracker.track_img('media\\img1.jpeg')
    # density = YoloTrackerAnalyzer.calc_density(w, h, outputs, 10, 10)


