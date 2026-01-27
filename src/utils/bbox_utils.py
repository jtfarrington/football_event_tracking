def get_center_of_bbox(bbox):
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
              (x1, y1) = top-left corner
              (x2, y2) = bottom-right corner
    
    Returns:
        Tuple of (x_center, y_center) as integers
    
    Example:
        bbox = [100, 200, 300, 400]
        center = get_center_of_bbox(bbox)  # Returns (200, 300)
    """
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
    
    Returns:
        Width in pixels (integer)
    
    Example:
        bbox = [100, 200, 300, 400]
        width = get_bbox_width(bbox)  # Returns 200
    """
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Uses the Pythagorean theorem: distance = sqrt((x2-x1)² + (y2-y1)²)
    
    Args:
        p1: First point as (x, y) tuple
        p2: Second point as (x, y) tuple
    
    Returns:
        Distance as a float
    
    Example:
        p1 = (0, 0)
        p2 = (3, 4)
        distance = measure_distance(p1, p2)  # Returns 5.0
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def measure_xy_distance(p1, p2):
    """
    Calculate X and Y distance components between two points.
    
    Returns how far apart the points are horizontally and vertically.
    Positive X = p1 is to the right of p2
    Positive Y = p1 is below p2
    
    Args:
        p1: First point as (x, y) tuple
        p2: Second point as (x, y) tuple
    
    Returns:
        Tuple of (x_distance, y_distance)
    
    Example:
        p1 = (100, 200)
        p2 = (50, 150)
        x_dist, y_dist = measure_xy_distance(p1, p2)  # Returns (50, 50)
    """
    x_distance = p1[0] - p2[0]
    y_distance = p1[1] - p2[1]
    return x_distance, y_distance


def get_foot_position(bbox):
    """
    Get the position where a player's feet touch the ground.
    
    Returns the bottom-center point of the bounding box.
    This represents where the player is standing on the field.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
    
    Returns:
        Tuple of (x_center, y_bottom) as integers
    
    Example:
        bbox = [100, 200, 300, 400]
        foot_pos = get_foot_position(bbox)  # Returns (200, 400)
    """
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_bottom = int(y2)
    return x_center, y_bottom