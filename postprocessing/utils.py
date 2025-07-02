from utils import get_y_center, get_x_center, load_vehicle_cat_config
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from typing import List, Tuple, Union, Dict


def get_center_points(items: List[Tuple[List[Union[int, float]], str]]) -> List[Tuple[str, List[float]]]:
    """
    Calculates the center (x, y) point for each text region in the list.

    Args:
        items (List[Tuple[bbox, str]]): List of tuples containing bounding boxes and text labels.

    Returns:
        List[Tuple[str, List[float]]]: List of text labels with their corresponding center coordinates.
    """
    centers_list = []

    for bbox, text in items:
        x_center = get_x_center(bbox)
        y_center = get_y_center(bbox)
        centers_list.append((text, [x_center, y_center]))

    return centers_list


def sort_by_category(items: List[Tuple[str, List[float]]]) -> List[Tuple[str, List[float]]]:
    """
    Sorts category items based on a predefined vehicle category order.

    Args:
        items (List[Tuple[str, List[float]]]): List of category labels with coordinates.

    Returns:
        List[Tuple[str, List[float]]]: Sorted list based on vehicle category priority.
    """
    category_order = load_vehicle_cat_config(is_to_sort=True)
    category_priority = {cat: i for i, cat in enumerate(category_order)}
    
    sorted_items = sorted(items, key=lambda x: category_priority.get(x[0], float('inf')))
    return sorted_items


def get_category_centers_list(items: List[Tuple[str, List[float]]]) -> Tuple[List[float], List[float]]:
    """
    Extracts separate lists of x and y center coordinates from category data.

    Args:
        items (List[Tuple[str, List[float]]]): List of category labels and center coordinates.

    Returns:
        Tuple[List[float], List[float]]: Lists of x and y center coordinates respectively.
    """
    x_centers = [coords[0] for _, coords in items]
    y_centers = [coords[1] for _, coords in items]
    return x_centers, y_centers


def get_adjecent_difference_sum(values: List[float]) -> float:
    """
    Calculates the sum of absolute differences between adjacent elements.

    Args:
        values (List[float]): A list of numeric values.

    Returns:
        float: Sum of absolute differences between adjacent elements.
    """
    total = 0.0
    for i in range(len(values) - 1):
        total += abs(values[i + 1] - values[i])
    return total


def deduplicate_categories(category_list, index):

    # Step 1: Group all values by label
    label_to_coords = defaultdict(list)
    for label, coords in category_list:
        label_to_coords[label].append(coords)

    # Step 2: Compute mean using only labels that appear once
    unique_values = [
        coords[index] for label, coords_list in label_to_coords.items()
        if len(coords_list) == 1
        for coords in coords_list
    ]
    mean_ = np.mean(unique_values)

    # Step 3: For duplicates, keep the one with closest to mean_
    final_coords = {}
    for label, coords_list in label_to_coords.items():
        if len(coords_list) == 1:
            final_coords[label] = coords_list[0]
        else:
            # Choose the one with smallest distance to mean_
            closest_coords = min(coords_list, key=lambda c: abs(c[index] - mean_))
            final_coords[label] = closest_coords

    # Step 4: Return as list
    return list(final_coords.items())


def complete_categories(category_list: List[Tuple[str, List[float]]]) -> List[Tuple[str, List[float]]]:
    """
    Fills in missing vehicle category coordinates by performing linear interpolation and extrapolation.

    Args:
        category_list (List[Tuple[str, List[float]]]): A list of tuples where each tuple contains a category label and its (x, y) center coordinates.

    Returns:
        List[Tuple[str, List[float]]]:A list of all expected vehicle categories with interpolated or extrapolated center coordinates.

    Raises:
        RuntimeError: If any error occurs during the interpolation or extrapolation process, such as issues with input data or interpolation failure.
    """
    try:
        # Get predefined order of all category labels
        cat_order: List[str] = load_vehicle_cat_config(is_to_sort=True)

        # Map input labels to their coordinates
        label_to_coords: dict[str, List[float]] = {label: coords for label, coords in category_list}
        label_indices: dict[str, int] = {label: i for i, label in enumerate(cat_order)}

        # Extract known indices and coordinates
        known_indices: List[int] = [label_indices[label] for label in label_to_coords]
        known_coords: np.ndarray = np.array([label_to_coords[label] for label in label_to_coords])
        known_x: np.ndarray = known_coords[:, 0]
        known_y: np.ndarray = known_coords[:, 1]

        # Create interpolation functions for x and y with extrapolation
        interp_x_func = interp1d(known_indices, known_x, kind='linear', fill_value='extrapolate')
        interp_y_func = interp1d(known_indices, known_y, kind='linear', fill_value='extrapolate')

        # Generate coordinates for all labels using interpolation/extrapolation
        full_list: List[Tuple[str, List[float]]] = []
        for i, label in enumerate(cat_order):
            x = float(interp_x_func(i))
            y = float(interp_y_func(i))
            full_list.append((label, [round(x, 2), round(y, 2)]))

        return full_list

    except Exception as e:
        raise RuntimeError(f"Error in complete_categories: {e}")


def get_bias(category_list: List[Tuple[str, List[float]]], orientation: str) -> int:
    """
    Determines the directional bias of the category layout.

    Args:
        category_list (List[Tuple[str, List[float]]]): List of vehicle categories with center coordinates.
        orientation (str): Either 'portrait' or 'landscape'.

    Returns:
        int: 1 if increasing order along axis, else 0.
    """
    idx = 1 if orientation == 'landscape' else 0
    return 1 if category_list[-1][1][idx] - category_list[0][1][idx] > 0 else 0


def get_dates_center(dates_centers_list: List[Tuple[str, List[float]]], orientation: str) -> float:
    """
    Calculates the mean center position of the date items.

    Args:
        dates_centers_list (List[Tuple[str, List[float]]]): 
            List of date text items with their center coordinates.
        orientation (str): 
            Either 'portrait' or 'landscape'.

    Returns:
        float: Average coordinate along the relevant axis.
    """
    idx = 0 if orientation == 'landscape' else 1
    total = sum(bbox[idx] for _, bbox in dates_centers_list)
    return total / len(dates_centers_list) if dates_centers_list else 0.0


def categorize_dates(dates_centers_list: List[Tuple[str, List[float]]], dates_center: float, orientation: str, bias: int) -> Tuple[List[Tuple[str, List[float]]], List[Tuple[str, List[float]]]]:
    """
    Divides dates into two categories based on their position and orientation.

    Args:
        dates_centers_list (List[Tuple[str, List[float]]]): 
            List of date items and their center positions.
        dates_center (float): 
            Central position separating the two date groups.
        orientation (str): 
            'portrait' or 'landscape'.
        bias (int): 
            Directional preference.

    Returns:
        Tuple[List, List]: Two categorized date lists.
    """
    idx = 0 if orientation == 'landscape' else 1
    category1, category2 = [], []

    for item in dates_centers_list:
        if item[1][idx] >= dates_center:
            category2.append(item)
        else:
            category1.append(item)

    if (bias == 1 and orientation == 'landscape') or (bias == 0 and orientation == 'portrait'):
        return category1, category2
    else:
        return category2, category1


def get_date_pairs(date_centers_list: List[Tuple[str, List[float]]], issued_dates: List[Tuple[str, List[float]]], expiry_dates: List[Tuple[str, List[float]]], orientation: str) -> Tuple[List[Tuple[Tuple[str, List[float]], Tuple[str, List[float]]]], List[Tuple[str, List[float]]]]:
    """
    Matches issued and expiry dates based on positional proximity.

    Args:
        date_centers_list (List[Tuple[str, List[float]]]): 
            All date items with centers.
        issued_dates (List[Tuple[str, List[float]]]): 
            Issued dates with centers.
        expiry_dates (List[Tuple[str, List[float]]]): 
            Expiry dates with centers.
        orientation (str): 
            'portrait' or 'landscape'.

    Returns:
        Tuple[List of matched pairs, List of unmatched items]
    """
    idx = 1 if orientation == 'landscape' else 0
    pairs_dict: Dict[Tuple[str, Tuple[float, float]], Tuple[str, List[float]]] = {}
    pairs_list = []
    submitted_dates = []

    # Ensure dates in pairs have closest neighbor in the pair
    for issued_date in issued_dates:
        pos_i = issued_date[1][idx]
        closest = min(expiry_dates, key=lambda e: abs(e[1][idx] - pos_i))
        pairs_dict[(issued_date[0], tuple(issued_date[1]))] = closest

    for exp_date in expiry_dates:
        pos_e = exp_date[1][idx]
        closest_i = min(issued_dates, key=lambda i: abs(i[1][idx] - pos_e))
        if pairs_dict.get((closest_i[0], tuple(closest_i[1]))) == exp_date:
            pairs_list.append((closest_i, exp_date))
            submitted_dates.extend([closest_i, exp_date])

    # Get dates not in pairs
    individual_dates = [item for item in date_centers_list if item not in submitted_dates]
    return pairs_list, individual_dates

    
def get_approx_category_position(pair: Tuple[Tuple[str, List[float]], Tuple[str, List[float]]], orientation: str) -> Tuple[float, int]:
    """
    Estimates category label position from date pair.

    Args:
        pair (Tuple): Issued and expiry date pair.
        orientation (str): 'portrait' or 'landscape'.

    Returns:
        Tuple[float, int]: Estimated position and axis index.
    """
    idx = 1 if orientation == 'landscape' else 0
    difference = pair[1][1][idx] - pair[0][1][idx]
    approx_pos = pair[0][1][idx] - difference
    return approx_pos, idx


def get_rows(category_list: List[Tuple[str, List[float]]], pairs_list: List[Tuple[Tuple[str, List[float]], Tuple[str, List[float]]]], orientation: str) -> Dict[str, List[str]]:
    """
    Assigns each date pair to the closest category.

    Args:
        category_list (List[Tuple[str, List[float]]]): List of category names with coordinates.
        pairs_list (List[Tuple]): List of issued and expiry date pairs.
        orientation (str): 'portrait' or 'landscape'.

    Returns:
        Dict[str, List[str]]: Mapping of category to [issued, expiry] texts.
    """
    rows_dict: Dict[str, List[str]] = {}

    for pair in pairs_list:
        approx_position, idx = get_approx_category_position(pair, orientation)
        closest = min(category_list, key=lambda c: abs(c[1][idx] - approx_position))
        rows_dict[closest[0]] = [pair[0][0], pair[1][0]]

    return rows_dict


def order_rows(rows: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Orders the dictionary of rows based on vehicle category config.

    Args:
        rows (Dict[str, List[str]]): Unordered mapping of category to date values.

    Returns:
        Dict[str, List[str]]: Ordered mapping based on predefined category order.
    """
    custom_order = load_vehicle_cat_config(True)
    return {key: rows[key] for key in custom_order if key in rows}

    
