import re
from typing import List, Tuple, Any
from utils import load_vehicle_cat_config, load_ocr_text_thresh_config


def validate_vehicle_categories(filtered_list: List[Tuple[Any, str, str]]) -> List[Tuple[Any, str]]:
    """
    Validate and clean OCR-detected vehicle category labels.

    Args:
        filtered_list (List[Tuple[bbox, text, category]]): OCR-extracted items possibly containing vehicle categories.
        vehicle_categories (List[str]): Valid vehicle category list.

    Returns:
        List[Tuple[bbox, cleaned_text]]: Validated category results.

    Raises:
        None
    """
    final_category_list: List[Tuple[Any, str]] = []

    for bbox, text, cat in filtered_list:
        # Exact match
        if text.strip() == cat:
            final_category_list.append((bbox, text))

        # Table line misinterpretation (e.g., 'IDE' -> 'DE')
        elif len(text) == 3 and text[1:] == cat and text[0] in {'I', 'i', '1'}:
            final_category_list.append((bbox, text[1:]))

        # Icon misinterpretation
        elif text[:2] == cat:
            final_category_list.append((bbox, text[:2]))

    return final_category_list


def validate_dates(filtered_dates: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
    """
    Validate and format OCR-detected dates to DD.MM.YYYY if appropriate.

    Args:
        filtered_dates (List[Tuple[bbox, date_text]]): OCR-extracted possible date strings.

    Returns:
        List[Tuple[bbox, formatted_date]]: Validated date results.

    Raises:
        None
    """
    final_dates_list: List[Tuple[Any, str]] = []

    for bbox, date in filtered_dates:
        is_valid = True

        # Remove detected texts with unwanted patterns to dates
        for pattern in {'..', '-', '_', '/'}:
            if pattern in date:
                is_valid = False
                break

        if not is_valid:
            continue

        concat_date = date.replace('.', '')

        # Check for correct date format (format if in correct form. If not pass the text without formattings)
        if len(concat_date) == 8 and not any(c.isalpha() for c in concat_date):
            formatted_date = concat_date[:2] + '.' + concat_date[2:4] + '.' + concat_date[4:]
            final_dates_list.append((bbox, formatted_date))
        else:
            final_dates_list.append((bbox, date))

    return final_dates_list


def extract_required_text_fields(ocr_results: List[Any]) -> Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]]]:
    """
    Extract vehicle categories and valid dates from OCR results based on confidence and pattern rules.

    Args:
        ocr_results (List[Any]): Raw OCR output from PaddleOCR.

    Returns:
        Tuple:
            - List[Tuple[bbox, category]]: Validated vehicle category texts.
            - List[Tuple[bbox, date]]: Validated date texts.

    Raises:
        RuntimeError: If config files cannot be loaded.
        ValueError: If ocr results are not in format.
    """

    filtered_categories: List[Tuple[Any, str, str]] = []
    filtered_dates: List[Tuple[Any, str]] = []

    try:
        # Load from config.yaml
        vehicle_categories = load_vehicle_cat_config(is_to_sort=False)
        confidence_threshold = load_ocr_text_thresh_config()

    except Exception as e:
        raise RuntimeError(f"Failed to load configuration files: {e}")
    

    # Seperate dates and category values
    for detection in ocr_results[0]:
        try:
            bbox, (text, confidence) = detection
        except Exception:
            raise ValueError(f"Malformed outputs from model: {e}")

        if confidence > confidence_threshold:
            text_clean = text.strip()

            # Primary category detection
            if len(text_clean) <= 5:
                for cat in vehicle_categories:
                    if cat in text_clean and cat != 'CE':
                        filtered_categories.append((bbox, text_clean, cat))
                        break

            # Primary date detection
            elif not re.search(r'(.*[a-zA-Z].*){3,}', text_clean):
                filtered_dates.append((bbox, text_clean))

    # Validate and filter with additional criteria
    categories = validate_vehicle_categories(filtered_categories)
    dates = validate_dates(filtered_dates)

    return categories, dates





