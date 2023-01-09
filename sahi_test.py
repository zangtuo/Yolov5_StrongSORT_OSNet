from sahi.utils.yolov5 import download_yolov5s6_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
import os.path as Path

from yolov5.models.common import DetectMultiBackend

# yolov5_model_path = 'yolov5n.pt'
yolov5_model_path = 'models/yolov5s6.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg'); download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)

model = DetectMultiBackend(yolov5_model_path, device='cpu', dnn=False, data=None, fp16=False)
result = get_prediction("demo_data/small-vehicles1.jpeg", model)

result.export_visuals(export_dir="demo_data/")
Image("demo_data/prediction_visual.png")

