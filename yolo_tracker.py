import numpy as np
import os
from pathlib import Path

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)

from trackers.multi_tracker_zoo import create_tracker

yolo_weights = Path('visdrone.pt')
tracking_method = 'osnet'

class YoloTracker:
    def __int__(self, yolo_weights, track_method, device, slice_enabled=False):
        self.device = device
        self.detection_model = DetectMultiBackend(yolo_weights, device=self.device, dnn=False, data=None, fp16=False)
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        self.imgsz = (640,640)
        self.imgsz = check_img_size(self.imgsz, s=stride)
        self.track_method = track_method
        self.reid_weights = None
        self.track_config = None


    '''监控车辆追踪'''
    def track_webcam_vehicle(self, source, frames):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        dataset = LoadStreams(source, img_size=self.imgsz, stride=stride, auto=pt, vid_stride=1)
        tracker = create_tracker(tracking_method, self.tracking_config, self.reid_weights, self.device, False)

        track_result = []
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if frame_idx > frames:
                return track_result






    '''无人机车辆追踪'''
    def track_vid_drone_vehicle(self, source):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)

    '''人员追踪'''
    def track_vid_pedestrian(self, source):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)

    '''人员追踪'''
    def track_webcam_pedestrian(self, source):
        stride, names, pt = self.detection_model.stride, self.detection_model.names, self.detection_model.pt
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride, auto=pt)


class YoloTrackerAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def calc_density(w, h, detection, step_x, step_y):
        grid_w = w / step_x
        grid_h = h / step_y
        density = np.zeros([step_x, step_y])
        for d in detection:
            centroid = [(d[0] + d[2])/2, (d[1] + d[3])/2, ]
            [x,y] = centroid
            idx_x = int(np.floor(x/grid_w))
            idx_y = int(np.floor(y/grid_h))
            density[idx_x, idx_y] += 1

        return density



if __name__ == '__main__':
    YoloTrackerAnalyzer.calc_density(100, 100, [[0.1, 0.1, 2.0, 2.0]], 10, 10)
    a = np.zeros((2,2))

    for m in range(2):
        for n in range(2):
            a[m,n] = m*2+n

    print(a[0,1])