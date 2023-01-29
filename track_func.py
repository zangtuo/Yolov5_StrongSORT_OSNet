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


# class YoloTracker:
#     def __int__(self, yolo_weights):
#         self.device = select_device('')
#         self.model = DetectMultiBackend(yolo_weights, device=self.device, dnn=False, data=None, fp16=False)
#
#     def track_vid_vehicle(self, vid):
#         pass
#
#     def track_vid_pedestrian(self, vid):
#         pass
#
#     def track_webcam_pedestrian(self, webcam):
#         self.tracker = create_tracker('bytetrack', )
#         pass
#
#     def


@torch.no_grad()
def track(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5n.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='bytetrack',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download, TODO: save into specific directories

    # TODO: define save_dir for result saving
    file_name = source.lower().split('\\')[-1]
    if is_file:
        save_dir = os.path.join(ROOT, file_name)
    else:
        save_dir = None

    save_dir = Path(save_dir)
    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    track_result = []
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            if is_seg:
                pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            else:
                pred = model(im, augment=augment, visualize=False)  # not visualize as service

        # Apply NMS
        with dt[2]:
            if is_seg:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            else:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                 im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                 im0.shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
            else:
                pass
                # tracker_list[i].tracker.pred_n_update_all_tracks()

            # Save results (image with detections)
            if save_vid:
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
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

        track_result.extend(outputs)

    return track_result


'''图像人群分析'''


def track_img(source='img.jpeg', h=1024, w=960, save_result=False):
    pass


'''监控设备'''


def track_webcam(url='rtsp://11.11.11.11', h=1024, w=960):
    # track_method = 'bytetrack'
    # track_config = ROOT / 'trackers' / track_method / 'configs' / (track_method + '.yaml')
    # track(yolo_weights=Path('yolov5n.pt'), source='media\\vid3.mp4', save_trajectories=True, tracking_config=track_config)
    pass


'''无人机视频车流分析'''


def track_vid_drone_vehicle(source='media/iiilab_video.mp4', h=1024, w=960):
    track_method = 'bytetrack'
    track_config = ROOT / 'trackers' / track_method / 'configs' / (track_method + '.yaml')

    # 检测结果# output的格式：[a, b, c, d, id, type, confidence], abcd是bounding box，
    # id是分类下的id，type是分类类型，confidence是置信
    result = track(yolo_weights=Path('visdrone.pt'), source=source, save_trajectories=True,
                   tracking_config=track_config)

    # 轨迹 + 速度计算：从最后一帧开始反向搜索type - id对应的目标bounding box质心(a + d) / 2, (b + c) / 2，
    # （最后一帧 - 第一次检测到目标帧 ） / 帧数差 = 速度向量
    # TODO: 处理track结果
    # 1. 绘制轨迹（基于类型+id：人或车辆）
    # draw(result, source)
    # 2. 计算每个车辆的帧移动速度 v = trajactory / frames
    frames_speed = speed(result)
    # 3. 计算网格区域车辆 num / grid
    grid_car(result)
    # 4. 计算区域密度变化趋势:

    #   a. 依据网格平均移动速度*时间 移动该网格内所有车辆至其他网格
    #   b. 重新计算网格车辆密度：如出现车辆聚集，则生成异常判决


def gpr2d(obs1, obs2, target_pos):
    obs1 = [[0, 0, 45], [0, 1, 34]]
    obs2 = [[0, 0, 45], [0, 1, 34]]
    target_pos[[0, 0], [0, 1]]

    return np.array([[0, 0, 45], [0, 1, 34]])


def density(img_url, grid_w, grid_h):
    return [
        [0, 0, 10],
        [0, 1, 100],
        [1, 1, 20]
    ]


def draw(result, source):
    """
    画车辆和行人的轨迹
    """
    # result[0][0] = [1890.3820437712752, 1785.0, 1940.0, 1895.1322314040058, 1, 4.0, 0.8935593]

    video = cv2.VideoCapture(source)
    fps = video.get(cv2.CAP_PROP_FPS)
    nums = 0
    while True:  # 视频播放
        ret, frame = video.read()
        if not ret:
            break
        # for j, (outputs) in enumerate(result[])
        if nums < 80:
            for num in range(nums):  # 画轨迹
                if num == 0:
                    continue
                for output in result[num]:
                    bbox = output[0:4]
                    id = int(output[4])
                    cls = int(output[5])
                    conf = output[6]
                    random.seed(id)
                    cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                               cls, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
        else:
            for num in range((nums - 80), nums):
                for output in result[num]:
                    bbox = output[0:4]
                    id = int(output[4])
                    cls = int(output[5])
                    conf = output[6]
                    random.seed(id)
                    cv2.circle(frame, (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)),
                               cls, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
        nums = nums + 1
        # x1 = result[70][0][0]
        # cv2.line(frame, (int(x1), int(100)), (int(700), int(700)), (255, 0, 0), 2, 8, 0)

        cv2.imshow(source, frame)  # 显示
        cv2.waitKey(int(300 / fps))
    cv2.destroyAllWindows()


def speed(result):
    """
    求result中所有类别的帧速度，但是暴力遍历
    return speeds:list
    """
    position = []
    speeds = []  # 速度向量
    id_list = []
    for frame, outputs in enumerate(result, 1):
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
                position.append([id, cls, start_frame, start_frame, start_pos_x, start_pos_y, start_pos_x, start_pos_y])
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


def grid_car(result, source='media/iiilab_video.mp4', grid_shape=[5, 5], color=(0, 255, 0)):
    """
    将视频分为grid_shape，求每个网格中的车辆数目，播放视频
    """
    list_x = [0]
    list_y = [0]

    video = cv2.VideoCapture(source)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret, frame = video.read()

    h, w, _ = frame.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(frame, (x, 0), (x, h), color=color, thickness=2)
        list_x.append(x)
    list_x.append(w)
    # 网格参数
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(frame, (0, y), (w, y), color=color, thickness=2)
        list_y.append(y)
    list_y.append(h)

    nums = 0
    while True:  # 视频播放
        car_number = np.zeros((rows, cols))  # 维度跟网格中的对应

        ret, frame = video.read()
        if not ret:
            break

        # 画网格
        for x in list_x:
            cv2.line(frame, (x, 0), (x, h), color=color, thickness=3)
        for y in list_y:
            cv2.line(frame, (0, y), (w, y), color=color, thickness=3)

        for output in result[nums]:  # 一帧中的所有目标
            if int(output[5]) == 4:
                bbox = output[0:4]
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                #  车辆计数
                for num_x, x in enumerate(list_x, 0):
                    if center_x >= x:
                        continue
                    else:
                        for num_y, y in enumerate(list_y, 0):
                            if center_y >= y:
                                continue
                            else:
                                car_number[num_y - 1][num_x - 1] = car_number[num_y - 1][num_x - 1] + 1
                                break
                        break

        #  画数字
        for i in range(0, len(list_x) - 1):
            for j in range(1, len(list_y)):
                cv2.putText(frame, str(car_number[j - 1][i]), (list_x[i], list_y[j]), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                            (100, 200, 200), 2)

        nums = nums + 1
        cv2.imshow(source, frame)  # 显示
        cv2.waitKey(int(300 / fps))
    cv2.destroyAllWindows()

    # return car_number  # 叠加后返回网格车辆数


#
#
# def main(opt):
#     check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
#
#
if __name__ == "__main__":
    track_vid_drone_vehicle()
    # grid_car()
