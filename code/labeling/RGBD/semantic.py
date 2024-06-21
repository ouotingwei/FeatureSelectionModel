from ultralytics import YOLO
import cv2 as cv

class yolov8_offical_semantic:
    def __init__(self, img_path):
        self.img_path = img_path
        self.model = YOLO('code/training/weight/yolov8n-seg.pt')

    def segmantic(self, feature_points):
        results = self.model(self.img_path)  # predict on an image

    def kp_semantic(self, uv_position_of_kp):
        pass
    
    def visualize(self):
        pass

        

class yolov8_ade20k_semantic:
    def __init__(self, img_path):
        self.img_path = img_path
        self.model = YOLO('code/training/weight/yolov8n-seg.pt')

    def segmantic(self):
        results = self.model(self.img_path)  # predict on an image

    def visualize(self):
        pass