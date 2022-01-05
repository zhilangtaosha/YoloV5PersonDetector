import datetime
import os
import queue
import threading
import time
import traceback
import torch
from cv2 import cv2
import warnings

warnings.filterwarnings("ignore")


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class FrameGetter:
    def __init__(self, rtsp_ip):
        self.rtsp_ip = rtsp_ip

        self.captured_video = None
        self.frame = None
        self.generated_frame = None
        self.cap = self.connect_to_camera()

    def connect_to_camera(self):
        self.captured_video = None
        try:
            self.captured_video = VideoCapture(self.rtsp_ip)
        except:
            print(traceback.print_exc())
        return self.captured_video

    def generate_frame(self, cap):
        self.generated_frame = None
        try:
            self.generated_frame = cap.read()
        except:
            print(f"[-] There was a problem during reading a frame from IP camera")
            print(traceback.print_exc())
        return self.generated_frame

    def get_frame_from_camera(self):
        self.frame = self.generate_frame(self.cap)
        if self.frame is not None:
            self.frame = cv2.resize(self.frame, (640, 380))
        return self.frame


class ModelClassSelector:
    # Official list - don't change it
    CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                   'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear',
                   'hair drier', 'toothbrush']

    def select_classes(self, selected_class_names_list=None, select_all_classes=False):
        _names = []
        if not select_all_classes or selected_class_names_list is not None:
            for list_object in selected_class_names_list:
                index_of_object = self.CLASS_NAMES.index(list_object)
                _names.append(index_of_object)
        else:
            for list_object in self.CLASS_NAMES:
                index_of_object = self.CLASS_NAMES.index(list_object)
                _names.append(index_of_object)
        return _names


class ModelConfigurator():
    def __init__(self, path_to_yolov5_folder, model_type, selected_classes, minimum_probability_for_detection=0.3):
        self.path_to_yolov5_folder = path_to_yolov5_folder
        self.model_type = model_type
        self.selected_classes = selected_classes
        self.minimum_probability_for_detection = minimum_probability_for_detection


class ImageManager:
    @staticmethod
    def save_image(camera_name, frame):
        image_counter = 0
        while True:
            if os.path.exists(f"{camera_name}\\{camera_name}_{image_counter}.jpg"):
                image_counter += 1
            else:
                break
        cv2.imwrite(f"{camera_name}\\{camera_name}_{image_counter}.jpg", frame)


class DetectionPerformer:
    @staticmethod
    def perform_detection_from_camera_frame(camera_name, detection_model, frame_getter):
        frame_from_camera = frame_getter.get_frame_from_camera()
        results = detection_model(frame_from_camera)
        model_has_detected_object = len(results.get_percentage_results()) > 0
        if model_has_detected_object:
            ImageManager.save_image(camera_name, frame_from_camera)


if __name__ == "__main__":
    class_selector = ModelClassSelector()
    selected_classes_for_detection = class_selector.select_classes(["person", "car", "truck"])

    model_config = ModelConfigurator(r'yolov5-6.0',
                                     'yolov5s',
                                     selected_classes_for_detection,
                                     minimum_probability_for_detection=0.3)
    print("Loading model..")
    model = torch.hub.load(model_config.path_to_yolov5_folder, model_config.model_type, pretrained=True, source="local")
    model.classes = model_config.selected_classes
    model.conf = model_config.minimum_probability_for_detection
    print("Model loaded")
    # model.iou = 0.25 # zapobieganie nakładaniu się obiektów
    print("Connecting to cameras...")
    frame_getter_sprzedaz3 = FrameGetter("rtsp://admin:Admin5487!@192.168.10.64:554/")
    frame_getter_sprzedaz2 = FrameGetter("rtsp://admin:Admin5487!@192.168.10.63:554/")
    frame_getter_sprzedaz1 = FrameGetter("rtsp://admin:Admin5487@192.168.10.58:554/")
    frame_getter_hala1 = FrameGetter("rtsp://admin:Ab-6472391614@192.168.10.103:554/")
    frame_getter_droga = FrameGetter("rtsp://admin:Ab-6472391614@192.168.10.51:554/")
    print("Connected successfully")
    camera_names = ["sprzedaz3", "sprzedaz2", "sprzedaz1", "hala1", "droga_obok_h1"]
    for camera_name in camera_names:
        if not os.path.isdir(camera_name):
            os.mkdir(camera_name)
    print("Starting detection...")
    while True:
        full_loop_time = time.time()
        DetectionPerformer.perform_detection_from_camera_frame(camera_names[0], model, frame_getter_sprzedaz3)
        DetectionPerformer.perform_detection_from_camera_frame(camera_names[1], model, frame_getter_sprzedaz2)
        DetectionPerformer.perform_detection_from_camera_frame(camera_names[2], model, frame_getter_sprzedaz1)
        DetectionPerformer.perform_detection_from_camera_frame(camera_names[3], model, frame_getter_hala1)
        DetectionPerformer.perform_detection_from_camera_frame(camera_names[4], model, frame_getter_droga)
        print("FULL DETECTION LOOP IN:", time.time() - full_loop_time,"seconds")
