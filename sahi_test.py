import os
os.getcwd()

# arrange an instance segmentation model for test
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# download YOLOV5S6 model to 'models/yolov5s6.pt'
# yolov5_model_path = 'models/yolov5s6.pt'
# download_yolov5s6_model(destination_path=yolov5_model_path)
yolov5_model_path = 'visdrone.pt'

# download test images into demo_data folder
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cuda:0", # or 'cuda:0'
)

result = get_prediction("media/test_drone.png", detection_model)

# result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)

result.export_visuals(export_dir="media/", file_name='prediction_yolo')

Image("media/prediction_visual.png")

result = get_sliced_prediction(
    "media/test_drone.png",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

result.export_visuals(export_dir="media/")

Image("media/prediction_visual.png")

object_prediction_list = result.object_prediction_list

object_prediction_list[0]