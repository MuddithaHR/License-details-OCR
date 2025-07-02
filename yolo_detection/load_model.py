from ultralytics import YOLO
import os

# def load_model(weights_path):
#     """
#     Load a YOLO model from the specified weights.

#     Args:
#         weights_path (str): Path to the YOLO model weights file.
   
#     Returns:
#         model (YOLO): Loaded YOLO model.
#     """

#     model = YOLO(weights_path)

#     return model



def load_model(weights_path: str) -> YOLO:
    """
    Load a YOLO model from the specified weights file.

    Args:
        weights_path (str): Path to the YOLO model weights file.
   
    Returns:
        model (YOLO): Loaded YOLO model.

    Raises:
        RuntimeError: If the model loading fails.
        FileNotFoundError : If weights path doesn't exist
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    

    try:
        #load yolo model
        model = YOLO(weights_path)
        return model
    
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model from '{weights_path}': {e}")

