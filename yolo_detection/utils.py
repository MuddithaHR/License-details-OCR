from utils import get_max_min_x_y_for_points_array
import numpy as np
from typing import List, Tuple, Union
from ultralytics.engine.results import Results

def crop_bounding_box(image: np.ndarray, box: List[float]) -> np.ndarray:
    """
    Crops a rectangular region from the input image based on the given bounding box.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        box (List[float]): Bounding box coordinates as [x1, y1, x2, y2].

    Returns:
        np.ndarray: Cropped image region.

    Raises:
        ValueError: If cropping fails due to invalid box or image shape.
    """

    margin = 3

    try:
        # Get margins of cropping image
        x1, y1, x2, y2 = get_max_min_x_y_for_points_array(box, margin, image.shape)
        crop = image[y1:y2, x1:x2]

        return crop
    
    except Exception as e:
        raise ValueError(f"Cropping failed due to unexpected error: {e}")


# def crop_bounding_box(image, box):
#     """
#     Crop a region from the image based on bounding box.

#     Args:
#         image (np.array): Input image.
#         boxes (list): Vertices list of bounding box (x1, y1, x2, y2).

#     Returns:
#         crops (list): cropped image region.
#     """

#     margin = 3

#     x1, y1, x2, y2 = get_max_min_x_y_for_points_array(box, margin, image.shape)
#     crops = image[y1:y2, x1:x2]

#     return crops


def get_chart_bounding_box(results: List[Results]) -> Tuple[Union[np.ndarray, None], np.ndarray, bool]:
    """
    Extracts the highest-confidence bounding box for class ID 0 ('chart') from YOLO detection results.

    A bounding box is only returned if its confidence score is greater than 0.85.

    Args:
        results (List[Results]): YOLO detection results list.

    Returns:
        Tuple:
            - Union[np.ndarray, None]: Bounding box [x1, y1, x2, y2] if found, else None.
            - np.ndarray: Original input image.
            - bool: True if a valid bounding box was found, else False.

    Raises:
        ValueError: If detection processing fails unexpectedly.
    """

    try:
        result = results[0]
        img_np = result.orig_img
        boxes = result.boxes

        # Get the bounding box with highest confidence (in more boxes are predicted)
        max_conf = 0.0
        max_conf_xyxy = None
        is_bbox_available = True

        for box in boxes:
            xyxy = box.xyxy.cpu().numpy()[0]    # Bounding box [x1, y1, x2, y2]
            conf = box.conf.cpu().numpy()[0]    # Confidence score
            cls = box.cls.cpu().numpy()[0]      # Class ID

            if cls == 0 and conf > max_conf:
                max_conf = conf
                max_conf_xyxy = xyxy


        # Check the confidence of best prediction
        if max_conf < 0.85:
            is_bbox_available = False
            max_conf_xyxy = None

        return max_conf_xyxy, img_np, is_bbox_available

    except Exception as e:
        raise ValueError(f"Chart bounding box detection failed: {e}")



# def get_chart_bounding_box(results):
#     """
#     Extract the highest-confidence bounding box for a 'chart' class from YOLO detection results. (with confidence > .85)

#     Args:
#         results (list): List of detection results from the YOLO model.

#     Returns:
#         max_conf_xyxy (np.array or None): Bounding box coordinates (x1, y1, x2, y2) with highest confidence
#                                            for class ID 0 (chart). Returns None if no detection meets the
#                                            confidence threshold (0.85).
#         img_np (np.array): The original input image as a NumPy array.
#         is_bbox_available (bool): whether valid bounding box available or not
#     """

    # result = results[0]

    # img_np = result.orig_img  # This is the original image as a NumPy array

    # boxes = result.boxes

    # max_conf = 0
    # max_conf_xyxy = None
    # is_bbox_available = True

    # for i, box in enumerate(boxes):
    #     # Extract box details bbox, confidence, classID
    #     xyxy = box.xyxy.cpu().numpy()[0]  
    #     conf = box.conf.cpu().numpy()[0]  
    #     cls = box.cls.cpu().numpy()[0]  

    #     if cls == 0 and conf > max_conf:
    #         max_conf = conf
    #         max_conf_xyxy = xyxy
        
    # if max_conf < .85:
    #     is_bbox_available = False
    #     max_conf_xyxy = None

    # return max_conf_xyxy, img_np, is_bbox_available