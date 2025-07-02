from .utils import get_chart_bounding_box, crop_bounding_box
import numpy as np
from ultralytics import YOLO
from typing import List, Union


# def detect_info_table(model, image_path):
#     """
#     Detect data table of the license in an image if the model confidence is > 0.85. Else, return whole image as a numpy array instead of cropped region.

#     Args:
#         mode (YOLO): loaded yolo model.
#         image_path (str): Path to input image.

#     Returns:
#         crops (list): List of cropped regions (np.arrays).
#     """

#     # Run inference
#     results = model(image_path)

#     #get bounding box of the table
#     bbox, image, is_bbox_available  = get_chart_bounding_box(results)

#     #crop the table if it is identified
#     if is_bbox_available:
#         crops = crop_bounding_box(image, bbox)
#     else:
#         crops = image

#     return crops


def detect_info_table(model: YOLO, image_path: str) -> Union[List[np.ndarray], np.ndarray]:
    """
    Detects a license data table in an image using a YOLO model. If the model detects the table 
    with a confidence score above .85, it returns the cropped region. 
    Otherwise, it returns the original image.

    Args:
        model (YOLO): Loaded YOLO model.
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Cropped region of the detected table or the original image.

    Raises:
        FileNotFoundError: If the image is not in the given path.
    """

    try:
        # Run YOLO inference
        results = model(image_path)

        # Get bounding box of the table 
        bbox, image_array, is_bbox_available = get_chart_bounding_box(results)

        # Crop or return full image
        if is_bbox_available:
            crop = crop_bounding_box(image_array, bbox)
            return crop
        else:
            return image_array

    except Exception as e:
        raise FileNotFoundError("Image not found at: " + image_path)