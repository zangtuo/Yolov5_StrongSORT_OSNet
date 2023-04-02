from Human import Hir_Yolo
import os

model_dir = '.'
model_name = 'yolov5x'


tracker = Hir_Yolo(yolo_weights=os.path.join(model_dir, model_name) + '.pt')
w, h, n_frame, outputs = tracker.interface_people_show(source='media_hir')