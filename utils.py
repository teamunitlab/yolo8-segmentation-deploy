import io
import urllib
from typing import List, Tuple, Union
import cv2
import numpy as np
from numpy import ndarray
from PIL import Image

W, H = 640, 640


# COCO Segmentation CLASSES
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


def get_image_from_url(url: str, rotation: int = 0) -> np.ndarray:
    """
    Fetches an image from a given URL, rotates it if specified, and returns it as an RGB NumPy array.

    Parameters:
    - url (str): The URL of the image to download.
    - rotation (int, optional): The angle to rotate the image by, in degrees. Defaults to 0 (no rotation).

    Returns:
    - numpy.ndarray: The downloaded and processed image as an RGB NumPy array.

    Raises:
    - ValueError: If the URL is invalid or the image cannot be processed.
    - Exception: For any unexpected errors during the image processing.
    """
    try:
        with urllib.request.urlopen(url) as fd:
            image_file = io.BytesIO(fd.read())

        with Image.open(image_file) as im:
            if rotation:
                # Rotate and expand the image to accommodate the new orientation
                im = im.rotate(rotation, expand=True)
            # Convert the image to RGB to ensure consistent format
            im = im.convert("RGB")

        # Convert the PIL image to a NumPy array for further processing
        image = np.array(im)
        return image

    except urllib.error.URLError as e:
        raise ValueError(f"Failed to download the image: {e.reason}")
    except Exception as e:
        # General exception handling to capture and raise unexpected errors
        raise Exception(f"An unexpected error occurred: {e}")


def download_image_from_url(url: str, coordinates: List = None, rotation: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads an image from a given URL, optionally rotates, crops to a specified rectangle, 
    and converts color space from RGB to BGR.

    :param url: URL of the image to be downloaded.
    :param coordinates: Optional tuple of four integers (x2, y2, x4, y4) defining the rectangle for cropping.
                        If None, the whole image is used.
    :param rotation: Degrees to rotate the image. Default is 0 (no rotation).
    :return: A tuple containing the original (or cropped if coordinates are provided) image and the color-converted image.
    """
    image = get_image_from_url(url, rotation)
    if coordinates is not None:
        try:
            x2, y2 = coordinates[:2]
            x4, y4 = coordinates[2:]
            x1, y1 = x2, y4
            x3, y3 = x4, y2
            poly = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            poly = np.array(poly).astype(np.int32).reshape(-1, 2)
            points = np.array([poly])
            rect = cv2.boundingRect(points)
            cropped_img = image[
                rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]
            ]
            image = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            return cropped_img, image
        except Exception as e:
            print(e)
    org_img = image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return org_img, image


def load_image_from_url(url: str, coordinates: List = None, rotation: int = 0, backend_module: str = "ONNX") -> Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load an image from a URL, optionally rotate and crop, convert to BGR, perform letterboxing, convert to RGB,
    prepare the image tensor for model input, and handle different backends.

    :param url: URL of the image.
    :param coordinates: A tuple of four integers defining the rectangle for cropping (optional).
    :param rotation: Degrees to rotate the image, default is 0.
    :param backend_module: The backend module used for further processing, default is "ONNX".
    :return: A tuple containing original image size, original BGR image, letterboxed BGR image,
             image tensor for model input, delta width and height for letterboxing, and scaling ratio.
    """
    _, bgr = download_image_from_url(url, coordinates, rotation)
    org_img = bgr.copy()
    size = bgr.shape
    bgr, ratio, dwdh = letterbox(bgr, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = np.array(dwdh * 2, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor)
    if backend_module == "ONNX":
        tensor = {'images': tensor}
    return size, org_img, bgr, tensor, dwdh, ratio


def letterbox(
        im: ndarray,
        new_shape: Union[Tuple, List] = (640, 640),
        color: Union[Tuple, List] = (114, 114, 114),
) -> Tuple[ndarray, float, Tuple[float, float]]:
    """
    Resizes and pads an image to a new shape, maintaining the aspect ratio and adding padding of a specific color.

    Parameters:
    - im: Input image as a NumPy ndarray.
    - new_shape: Desired image size as width, height. Can be a tuple or list.
    - color: Padding color, specified as a tuple or list of BGR values.

    Returns:
    - A tuple of:
        - Resized and padded image.
        - Scale ratio (new size / old size).
        - Width and height padding applied to the image.
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def blob(im: np.ndarray, return_seg: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares an image for neural network processing, optionally including a segmentation map.
    
    Parameters:
    - im: Input image as a NumPy ndarray with shape (height, width, channels).
    - return_seg: A boolean flag indicating whether to return a segmentation map along with the processed image.
    
    Returns:
    - If return_seg is False, returns a single ndarray representing the processed image with shape (1, channels, height, width).
    - If return_seg is True, returns a tuple of two ndarrays: the processed image and its segmentation map.
    """
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function over an input value or array.
    Parameters:
    - x: A scalar or Numpy array of any size.

    Returns:
    - The sigmoid of `x`, with the same shape as `x`.
    """
    return 1.0 / (1.0 + np.exp(-x))


def crop_mask(masks: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """
    Crops masks based on the provided bounding boxes.

    Parameters:
    - masks: A numpy array of shape (n, h, w), where n is the number of masks, h is the height, and w is the width.
    - bboxes: A numpy array of shape (n, 4), where each row contains the coordinates (x1, y1, x2, y2) of the bounding box for the corresponding mask.

    Returns:
    - A numpy array of the same shape as `masks`, where each mask is cropped to its corresponding bounding box, and areas outside the bounding box are set to 0.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(
        bboxes[:, :, None], [1, 2, 3], 1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def postprocess_segmentation(
    data: Tuple[np.ndarray, np.ndarray],
    downsample_width: int,
    downsample_height: int,
    image_shape: Tuple[int, int],
    target_labels: List[int],
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.65,
) -> List:
    """
    Post-process segmentation outputs to extract masks and contours for target labels.

    Parameters:
    - data: Tuple containing detection outputs and prototype masks.
    - downsample_width, downsample_height: Integers for downsampling dimensions.
    - image_shape: Tuple specifying the shape of the input image.
    - target_labels: List of target labels to filter results.
    - confidence_threshold: Float for the confidence score threshold.
    - iou_threshold: Float for the Intersection over Union threshold.

    Returns:
    - List of dictionaries with keys 'counters', 'mask_shape', and 'object_class'.
    """
    assert len(data) == 2
    h, w = image_shape[0] // 4, image_shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > confidence_threshold
    bboxes, scores, labels, maskconf = (
        bboxes[idx],
        scores[idx],
        labels[idx],
        maskconf[idx],
    )
    cvbboxes = np.concatenate(
        [bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], 1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split(".")[:2])
    assert v0 == 4, "OpenCV version is wrong"
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(
            cvbboxes, scores, labels, confidence_threshold, iou_threshold)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, confidence_threshold, iou_threshold)
    bboxes, scores, labels, maskconf = (
        bboxes[idx],
        scores[idx],
        labels[idx],
        maskconf[idx],
    )
    if len(bboxes) == 0:
        return []
    masks = sigmoid(maskconf @ proto).reshape(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.0)
    labels = labels + 1
    target_index = [idx for idx, label in enumerate(
        labels) if label in target_labels]
    masks = masks[target_index]
    scores = scores[target_index]
    pre_labels = labels[target_index]
    target_classes = [target_labels.index(inx) + 1 for inx in pre_labels]
    all_contours = []
    for idx, mask in enumerate(masks):
        object_class = target_classes[idx]
        mask = mask.reshape(160, 160)
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        mask = np.ascontiguousarray((mask > 0.5)[..., None], dtype=np.float32)
        mask = mask[downsample_height: 640 - downsample_height, downsample_width: 640 - downsample_width]
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(
            {"contours": contours, 'mask_shape': mask.shape, "object_class": int(object_class)})
    return all_contours
