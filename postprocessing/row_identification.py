from typing import List, Tuple, Dict
from .utils import (get_center_points,complete_categories,get_dates_center,get_bias,categorize_dates,get_date_pairs,get_rows,order_rows)


def identify_rows(
    dates_list: List[Tuple], orientation: str, category_centers_list: List[Tuple[str, List[float]]]) -> Tuple[str, Dict[str, Tuple[str, str]]]:
    """
    Identifies table rows by matching vehicle categories with issued and expiry dates based on spatial relationships and document orientation.

    Args:
        dates_list (List[Tuple[Tuple[int, int, int, int], str]]): List of detected date entries from OCR, each as (bounding box, text).
        orientation (str): Document layout orientation - either 'portrait' or 'landscape'.
        category_centers_list (List[Tuple[str, List[float]]]): List of vehicle category labels and their center coordinates.

    Returns:
        Tuple[str, Dict[str, Tuple[str, str]]]: 
            - A message indicating the detection status and a dictionary mapping
            - category labels to (issued_date, expiry_date) pairs.

    Raises:
        RuntimeError: If error occurs during execution.
    """
    try:
        # Get center points for dates
        date_centers_list = get_center_points(dates_list)

        # Ensure enough categories to predict rows
        if len(category_centers_list) <= 1:
            return 'Unable to identify categories properly.', {}

        # Interpolate and complete category layout
        completed_category_centers = complete_categories(category_centers_list)

        # Ensure enough date points for matching
        if len(date_centers_list) <= 1:
            return 'Unable to identify dates properly.', {}

        #Classify dates into issued and expiry using spatial bias
        dates_center = get_dates_center(date_centers_list, orientation)
        bias = get_bias(completed_category_centers, orientation)
        issued_dates, expiry_dates = categorize_dates( date_centers_list, dates_center, orientation, bias)

        # Form date pairs for each row
        date_pairs, unmatched_dates = get_date_pairs(date_centers_list, issued_dates, expiry_dates, orientation)


        # Combine category centers with matched date pairs
        if not date_pairs:
            return 'Unable to identify dates properly.', {}

        row_data = get_rows(completed_category_centers, date_pairs, orientation)
        sorted_rows = order_rows(row_data)

        # Validate completeness of output
        if len(sorted_rows) == len(date_pairs) and not unmatched_dates:
            return 'Detection Successful.', sorted_rows
        else:
            return 'Some rows are missing in the result.', sorted_rows

    except Exception as e:
        raise RuntimeError(f"Error during processing. {e}")



