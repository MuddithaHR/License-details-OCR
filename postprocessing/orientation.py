from typing import List, Tuple
from .utils import (get_center_points, sort_by_category, get_category_centers_list, get_adjecent_difference_sum)


def find_image_orientation(category_list: List[str]) -> Tuple[str, List[Tuple[str, Tuple[float, float]]]]:
    """
    Determines the orientation (portrait or landscape) of an image
    based on the alignment of category text center points.

    Args:
        category_list (List[str]): List of category names or detected text labels.

    Returns:
        Tuple[str, List[Tuple[str, Tuple[float, float]]]]:
            - Orientation as 'portrait' or 'landscape'.
            - Sorted list of tuples containing category names and their center coordinates.
    
    Raises:
        ValueError: If category list is empty or invalid.
    """

    # if not category_list:
    #     raise ValueError("Category list is empty or invalid.")

    # Get center points for each category text
    centers_list = get_center_points(category_list)

    # Sort centers by category label
    sorted_centers = sort_by_category(centers_list)

    # Extract separate lists for x and y center coordinates
    x_centers, y_centers = get_category_centers_list(sorted_centers)

    # Calculate adjacent difference sums for both x and y directions
    sum_x = get_adjecent_difference_sum(x_centers)
    sum_y = get_adjecent_difference_sum(y_centers)

    # Determine orientation: if variation along x-axis is higher, it's portrait (vertical list)
    orientation = 'portrait' if sum_x > sum_y else 'landscape'

    return orientation, sorted_centers







