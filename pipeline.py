from yolo_detection import load_model, detect_info_table
from utils import save_csv, load_yolo_weights_config
from ocr import load_ocr_model
from postprocessing import (extract_required_text_fields, find_image_orientation, identify_rows)
import cv2
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from typing import Any, Tuple


def load_models() -> Tuple[YOLO, PaddleOCR]:
    """
    Load both YOLO and OCR models, ensuring robust exception handling.

    Args:
        None
    
    Returns:
        tuple: A tuple containing the loaded YOLO model and PaddleOCR model.
    
    Raises:
        RuntimeError: If the loading of either the YOLO or OCR model fails.
    """
    try:
        # Load YOLO model 
        finetuned_weights_path = load_yolo_weights_config()
        yolo_model = load_model(finetuned_weights_path)

        # Load OCR model
        ocr_model = load_ocr_model()

        return yolo_model, ocr_model
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading models: {e}")
    



def detail_extraction_pipeline(yolo_model: YOLO, ocr_model: PaddleOCR, img_file_path: str) -> str:
    """
    Extract details from a license image using a YOLO model and OCR model.

    This function detects the information table from a license image, performs OCR on it,
    extracts necessary text fields (categories and dates), identifies the correct image
    orientation, pairs categories with dates, and saves the results to a CSV file.

    Args:
        yolo_model (YOLO): Pre-loaded YOLO object detection model.
        ocr_model (PaddleOCR): Pre-loaded OCR model.
        img_file_path (str): Path to the image file.

    Returns:
        str : Feedback message

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If detection or OCR fails at any stage.
    """


    if not os.path.exists(img_file_path):
        raise FileNotFoundError(f"Image file is not in the specified path: {img_file_path}")

    try:
        # Detect information table and crop
        crops = detect_info_table(yolo_model, img_file_path)

         # Convert to grayscale for better OCR
        crops = cv2.cvtColor(crops, cv2.COLOR_BGR2GRAY)

        # Step 2: Perform OCR on cropped image
        results = ocr_model.ocr(crops, cls=True)

        if bool(results[0]):

            # Extract dates and categories from OCR output
            categories, dates = extract_required_text_fields(results)

            # Determine orientation and category positions
            image_orientation, category_centers = find_image_orientation(categories)

            # Get category, date pairs of the license
            feedback_text, cat_date_pairs = identify_rows(dates, image_orientation, category_centers)

            # Save output to CSV if found
            if cat_date_pairs:
                save_csv(cat_date_pairs, img_file_path)

            return feedback_text

        else:
            return 'No output from OCR.'

    except Exception as e:
        raise RuntimeError(f"Failed to complete detail extraction pipeline: {e}")
