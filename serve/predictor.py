import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import CLASSES_GENERAL, load_image_from_url, postprocess_segmentation
import os


app = Flask(__name__)
cors = CORS(app)


BACKEND_MODULE = os.environ["BACKEND_MODULE"]



def get_engine_backend():
    if BACKEND_MODULE == 'ONNX':
        from modules.onnx_engine import ONNXModule
        EngineSegment = ONNXModule("weights/model.onnx")
        return EngineSegment
    elif BACKEND_MODULE == 'TENSORRT':
        from modules.trt_engine import TRTModule
        EngineSegment = TRTModule("weights/model.engine")
        return EngineSegment
    else:
        raise Exception("Please set up the BACKEND_MODULE environment variable as ONNX or TENSORRT") 


InferenceEngine = get_engine_backend()


@app.route("/api/yolo8-coco-segmentation", methods=["GET", "POST"])
def run_coco_segmentation():
    content = request.json
    image_source = content["src"]
    
    # Load and preprocess the image
    preprocessing_params = {
        "coordinates": content.get("coordinates"),
        "rotation": content.get("rotation", 0),
        "backend_module": BACKEND_MODULE,
    }
    size, original_image, background_removed_image, image_tensor, downscaled_width_height, scaling_ratio = load_image_from_url(
        image_source, **preprocessing_params
    )
    
    downscaled_width, downscaled_height =  int(downscaled_width_height[0]), int(downscaled_width_height[1])
    
    # Initialize inference engine and run segmentation
    inference_engine = InferenceEngine(image_tensor)
    coco_classes = range(0, 80)  # COCO dataset class indices (0-79)
    segmentation_thresholds = {"confidence_threshold": 0.25, "iou_threshold": 0.65}
    segmentation_results = postprocess_segmentation(
        inference_engine, downscaled_width, downscaled_height, background_removed_image.shape[:2],
        coco_classes, **segmentation_thresholds
    )   
    # Process segmentation results
    annotations = []
    predicted_classes = set()
    for result in segmentation_results:
        contours = result["contours"]
        object_class = result["object_class"]
        mask_shape = result["mask_shape"]
        for contour in contours:
            if cv2.contourArea(contour) < 500 or len(contour) < 3:
                continue  # Filter out small or invalid contours
            normalized_contour = contour / (
                mask_shape[0],
                mask_shape[1],
            )
            normalized_contour = normalized_contour * (size[0], size[1])
            point = normalized_contour.reshape((-1, 2))
            point = point.flatten()
            annotations.append(
                {
                    "segmentation": [np.round(point, decimals=1).tolist()],
                    "category_id": object_class - 1,
                }
            )
            predicted_classes.add(object_class - 1)
    segmentation_response = {"annotations": annotations,
              "predicted_classes": list(predicted_classes)}
    return jsonify(segmentation_response)


@app.route("/api/yolo8-person-segmentation", methods=["GET", "POST"])
def run_person_segmentation():
    content = request.json
    image_source = content["src"]
    
    # Load and preprocess the image
    preprocessing_params = {
        "coordinates": content.get("coordinates"),
        "rotation": content.get("rotation", 0),
        "backend_module": BACKEND_MODULE,
    }
    size, original_image, background_removed_image, image_tensor, downscaled_width_height, scaling_ratio = load_image_from_url(
        image_source, **preprocessing_params
    )
    
    downscaled_width, downscaled_height =  int(downscaled_width_height[0]), int(downscaled_width_height[1])
    
    # Initialize inference engine and run segmentation
    inference_engine = InferenceEngine(image_tensor)
    person_classes_from_coco = [1, 25, 27, 28]  
    segmentation_thresholds = {"confidence_threshold": 0.25, "iou_threshold": 0.65}
    segmentation_results = postprocess_segmentation(
        inference_engine, downscaled_width, downscaled_height, background_removed_image.shape[:2],
        person_classes_from_coco, **segmentation_thresholds
    )   
    # Process segmentation results
    annotations = []
    predicted_classes = set()
    for result in segmentation_results:
        contours = result["contours"]
        object_class = 1 # Define the Person class with only one class identifier
        mask_shape = result["mask_shape"]
        for contour in contours:
            if cv2.contourArea(contour) < 500 or len(contour) < 3:
                continue  # Filter out small or invalid contours
            normalized_contour = contour / (
                mask_shape[0],
                mask_shape[1],
            )
            normalized_contour = normalized_contour * (size[0], size[1])
            point = normalized_contour.reshape((-1, 2))
            point = point.flatten()
            annotations.append(
                {
                    "segmentation": [np.round(point, decimals=1).tolist()],
                    "category_id": object_class - 1,
                }
            )
            predicted_classes.add(object_class - 1)
    segmentation_response = {"annotations": annotations,
              "predicted_classes": list(predicted_classes)}
    return jsonify(segmentation_response)