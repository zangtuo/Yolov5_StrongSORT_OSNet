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
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)
        outputs = [None]
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())

        result = []
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

                # TODO sliced prediction
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

                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                 im0.shape).round()  # rescale boxes to im0 size

                        # pass detections to strongsort
                        with dt[3]:
                            outputs[i] = tracker.update(det.cpu(), im)

                result.append(outputs)

        return w, h, result

    '''视频追踪分析'''
    @torch.no_grad()
    def track_vid(self, source):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
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

                #TODO sliced prediction
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


        return (w,h,n_frame,track_result)

class YoloTrackerAnalyzer:
    def __init__(self):
        pass

    def calc_density(self, w, h, classes, grid_w, grid_h, result):
        density = np.zeros([grid_w, grid_h])
        for r in result:
            centroid = [(r[0] + r[2])/2, (r[1] + r[3])/2]
            [x,y] = centroid
            idx_x = int(np.floor(x/w*grid_w))
            idx_y = int(np.floor(y/h*grid_h))
            density[idx_x, idx_y] += 1

        return density

    '''calculate speed of each object based on the last n frames'''
    def calc_speed(self, result, classes, n_frame):
        res = result[-n_frame:]
        position = []
        speeds = []
        id_list = []
        for frame, outputs in enumerate(res, 1):
            for output in outputs:
                bbox = output[0:4]
                id = int(output[4])
                cls = int(output[5])
                if id not in id_list:
                    id_list.append(id)
                    start_frame = frame
                    start_pos_x = (bbox[0] + bbox[2]) / 2
                    start_pos_y = (bbox[1] + bbox[3]) / 2
                    # [id, cls, start_frame, end_frame, start_pos_x, start_pos_y, end_pos_x, end_pos_y]
                    position.append(
                        [id, cls, start_frame, start_frame, start_pos_x, start_pos_y, start_pos_x, start_pos_y])
                else:
                    position[id_list.index(id)][3] = frame
                    position[id_list.index(id)][6] = (bbox[0] + bbox[2]) / 2
                    position[id_list.index(id)][7] = (bbox[1] + bbox[3]) / 2

        for num in range(len(position)):  # 速度向量
            frame_d = position[num][3] - position[num][2]
            if frame_d == 0:
                speeds.append([position[num][0], position[num][1], 0, 0])
            else:
                id_speedx = (position[num][6] - position[num][4]) / frame_d
                id_speedy = (position[num][7] - position[num][5]) / frame_d
                # [id, type, x, y]
                speeds.append([position[num][0], position[num][1], id_speedx, id_speedy])

        return speeds

    def esti_trend(self, detection):
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        val = 0
        trend = [x1, y1, x2, y2, val]
        trends = [trend]
        return trends


    '''人群异常聚集检测'''
    def esti_agglo(self, w, h, grid_w, grid_h, results):
        threshold = 20
        density = np.array(self.calc_density(w, h, results, grid_w, grid_h))

        # compute mean, std, and media of density
        mu_d = np.average(density)
        std_d = np.std(density)
        median_d = np.median(density)

        # TODO if a grid's density is above the threshold && higher than 3*std_d+median_d, consider it as agglomerated

    '''人群异常逃离检测'''
    def esti_flee(self, w, h, grid_w, grid_h, results):
        density = []
        for result in results:
            dense = np.array(self.calc_density(w, h, result, grid_w, grid_h))
            density.append(dense)

        density = np.array(density)

        #TODO calculate

    '''人群冲突检测'''
    def esti_conflict(self, w, h, grid_w, grid_h, results):
        pass

'''结果可视化，结果保存'''
class YoloTrackerVisualizer:
    def draw_track(self, w, h, tracks):
        # tracks smoothing
        pass

    def draw_boundbox(self, w, h, result):
        pass

    def draw_vid(self, w, h, video, result, file_save, draw_bb=True, draw_track=False, track_frame=10):
        pass


if __name__ == '__main__':
    # hyper-parameters
    grid_w = 16
    grid_h = 16
    model_dir = '.'
    model_name = 'visdrone'
    classes = [3,4,5,6,7,8,9]

    analyzer = YoloTrackerAnalyzer()
    tracker = YoloTracker(yolo_weights=os.path.join(model_dir, model_name)+'.pt')

    w, h, n_frame, outputs = tracker.track_vid('media\\vid3.mp4')

    # '''density calculation'''
    density = analyzer.calc_density(w, h, grid_w, grid_h, classes, outputs[20][0])

    '''track draw based on the last frame of video'''
    # tracks = YoloTrackerAnalyzer.calc_last_track(w, h, outputs, frame_recur)
    # track_img = YoloTrackerAnalyzer.draw_last_track(w, h, outputs)

    # true_north = 100
    # YoloTrackerAnalyzer.calc_trend(w, h, outputs, true_north)

    # w, h, outputs = tracker.track_img('media\\img1.jpeg')
    # density = YoloTrackerAnalyzer.calc_density(w, h, outputs, 10, 10)


