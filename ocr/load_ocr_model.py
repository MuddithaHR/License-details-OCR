from paddleocr import PaddleOCR


def load_ocr_model() -> PaddleOCR:
    """
    Load a PaddleOCR model with English language support and angle classification enabled.

    Returns:
        ocr (PaddleOCR): Loaded PaddleOCR model instance.

    Raises:
        RuntimeError: If loading the PaddleOCR model fails.
    """
    try:
        # Attempt to load the OCR model
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        return ocr
    
    except Exception as e:
        raise RuntimeError(f"Failed to load PaddleOCR model: {e}")
