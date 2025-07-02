import yaml

def load_config(config_path="configs/config.yaml"):
    """
    Load configuration settings from a YAML file.

    Args:
        config_path (str): Path to the configuration file (default is "configs/config.yaml").

    Returns:
        config (dict): Parsed configuration settings from the YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_yolo_weights_config():
    """
    Load YOLO weights path.

    Returns:
        weights_path (str): Path to the YOLO model weights file..
    """
    config = load_config()
    weights_path = config['yolo_model']['weights_path']

    return weights_path


def load_yolo_thresh_config():
    """
    Load YOLO-specific configuration setting, confidence threshold.

    Returns:
        confidence_threshold (float): Confidence threshold for YOLO model predictions.
    """
    config = load_config()
    confidence_threshold = config['yolo_model']['conf_threshold']

    return confidence_threshold


def load_vehicle_cat_config(is_to_sort):
    """
    Load vehicle category constraints from the configuration.

    Args:
        is_to_sort (bool): whether asking for sorting purpose

    Returns:
        categories (list): A list of allowed vehicle categories defined in the constraints section.
    """
    config = load_config()

    if is_to_sort:
        categories = config['constraints']['vehicle_categories_for_sort']
    else:
        categories = config['constraints']['vehicle_categories_for_check'] 

    return categories


def load_ocr_text_thresh_config():
    """
    Load OCR-specific text confidence threshold from the configuration.

    Returns:
        ocr_text_threshold (float): Minimum confidence score required to accept OCR-detected text.
    """
    config = load_config()
    ocr_text_threshold = config['constraints']['ocr_text_threshold']

    return ocr_text_threshold


def load_output_path_config():
    """
    Load the output path to save CSV files.

    Returns:
        folder_path (str): Minimum confidence score required to accept OCR-detected text.
    """

    config = load_config()
    folder_path = config['output']['save_dir']

    return folder_path


