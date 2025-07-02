def get_max_min_x_y_for_points_array(arr, margin, img_shape):
    x1, y1, x2, y2 = map(int, arr)
    max_x = max(0, x1 - margin)
    max_y = max(0, y1 - margin)
    min_x = min(img_shape[1], x2 + margin)
    min_y = min(img_shape[0], y2 + margin)

    return max_x,max_y,min_x,min_y


# def get_max_min_x_y_for_coordinates_array(arr):
#     a, b, c, d = arr
#     max_x = max(a[0],b[0],c[0],d[0])
#     min_x = min(a[0],b[0],c[0],d[0])
#     max_y = max(a[1],b[1],c[1],d[1])
#     min_y = min(a[1],b[1],c[1],d[1])

#     return max_x, max_y, min_x, min_y


def get_y_center(box):
    ys = [point[1] for point in box]
    y1, y2 = min(ys), max(ys)
    return (y1+y2) / 2


def get_x_center(box):
    xs = [point[0] for point in box]
    x1, x2 = min(xs), max(xs)
    return (x1+x2) / 2


