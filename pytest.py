import json
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import load_image_from_url, seg_postprocess
import os


app = Flask(__name__)
cors = CORS(app)


from dotenv import dotenv_values
env_vars = dotenv_values(".env")
os.environ.update(env_vars)


BACKEND_MODULE = os.environ["BACKEND_MODULE"]


CLASSES_GENERAL = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def set_engine():
    if BACKEND_MODULE == 'ONNX':
        from modules.onnx_engine import ONNXModule
        EngineSegment = ONNXModule("weights/yolov8x-seg.onnx")
        return EngineSegment
    elif BACKEND_MODULE == 'TENSORRT':
        from modules.trt_engine import TRTModule
        EngineSegment = TRTModule("weights/yolov8x-seg.engine")
        return EngineSegment
    else:
        raise Exception("Please set up the BACKEND_MODULE environment variable as ONNX or TENSORRT") 
        

Engine = set_engine();
    

@app.route("/api/yolo8-coco-segmentation", methods=["GET", "POST"])
def run_coco_segmentation():
    content = request.json
    src = content["src"]
    size, org_img, brg, tensor, dwdh, ratio = load_image_from_url(
        src, content.get("coordinates", None), content.get("rotation", 0), BACKEND_MODULE
    )
    dw, dh = int(dwdh[0]), int(dwdh[1])
    data = Engine(tensor)
    target_labels = range(1, 80)
    targets = seg_postprocess(data, dw, dh, brg.shape[:2], target_labels, 0.25, 0.65)
    points = []
    predicted_classes = set()
    for target in targets:
        contours = target["counters"]
        object_class = 1
        mask_shape = target["mask_shape"]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            if len(contour) < 3:  # less than three cortexes can't not be polygon
                continue
            approx_contour = contour / (
                mask_shape[0],
                mask_shape[1],
            )
            approx_contour = approx_contour * (size[0], size[1])
            point = approx_contour.reshape((-1, 2))
            point = point.flatten()
            points.append(
                {
                    "segmentation": [np.round(point, decimals=1).tolist()],
                    "category_id": object_class - 1,
                }
            )
            predicted_classes.add(object_class - 1)
    result = {"annotations": points, "predicted_classes": list(predicted_classes)}
    return jsonify(result)

@app.route("/api/yolo8-coco-person-segmentation", methods=["GET", "POST"])
def run_person_segmentation():
    content = request.json
    src = content["src"]
    size, org_img, brg, tensor, dwdh, ratio = load_image_from_url(
        src, content.get("coordinates", None), content.get("rotation", 0), BACKEND_MODULE
    )
    dw, dh = int(dwdh[0]), int(dwdh[1])
    data = Engine(tensor)
    target_labels = [1, 25, 27, 28]
    targets = seg_postprocess(data, dw, dh, brg.shape[:2], target_labels, 0.25, 0.65)
    points = []
    predicted_classes = set()
    for target in targets:
        contours = target["counters"]
        object_class = 1
        mask_shape = target["mask_shape"]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            if len(contour) < 3:  # less than three cortexes can't not be polygon
                continue
            approx_contour = contour / (
                mask_shape[0],
                mask_shape[1],
            )
            approx_contour = approx_contour * (size[0], size[1])
            point = approx_contour.reshape((-1, 2))
            point = point.flatten()
            points.append(
                {
                    "segmentation": [np.round(point, decimals=1).tolist()],
                    "category_id": object_class - 1,
                }
            )
            predicted_classes.add(object_class - 1)
    result = {"annotations": points, "predicted_classes": list(predicted_classes)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=56044)