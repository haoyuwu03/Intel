from yolov3 import create_yolov3
from utils import load_yolo_weights, detect_image
from configurations import yolo_input_size, yolo_darknet_weights

input_size = yolo_input_size
darknet_weights = yolo_darknet_weights

image_path = "D:/YOLOv3/test.jpg"
output_path = "D:/YOLOv3"

yolo = create_yolov3(input_size)
load_yolo_weights = (yolo, darknet_weights)

detect_image(yolo, image_path, output_path = output_path, show = True, colab = False)