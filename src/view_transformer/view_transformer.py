import numpy as np 
import cv2


class ViewTransformer():
    """
    Transforms pixel coordinates from video to real-world field coordinates.
    
    Problem: Objects appear different sizes based on distance from camera.
    - Player at bottom of screen: appears larger, moves more pixels per meter
    - Player at top of screen: appears smaller, moves fewer pixels per meter
    
    Solution: Perspective transformation
    - Maps four points in video to their known real-world positions
    - Allows conversion from pixels → meters
    - Accounts for camera angle and distance
    
    This enables accurate speed/distance calculations in real-world units.
    """
    
    def __init__(self):
        """
        Initialize the perspective transformer with field dimensions and calibration points.
        
        Calibration: Four corner points of the visible field area are manually identified
        in the video and mapped to their real-world coordinates.
        
        Note: These pixel vertices are specific to the camera angle/position of your video.
        Different videos require different calibration points.
        """
        # Real-world field dimensions (in meters)
        # Standard football pitch segment visible in frame
        court_width = 68      # Width of pitch (touchline to touchline)
        court_length = 23.32  # Length of visible pitch segment

        # Pixel coordinates of four corners in the video frame
        # These are manually identified points (specific to this video)
        # Order: [bottom-left, top-left, top-right, bottom-right]
        self.pixel_vertices = np.array([
            [110, 1035],   # Bottom-left corner
            [265, 275],    # Top-left corner
            [910, 260],    # Top-right corner
            [1640, 915]    # Bottom-right corner
        ])
        
        # Real-world coordinates of those same four corners (in meters)
        # Origin (0,0) is at top-left corner
        self.target_vertices = np.array([
            [0, court_width],      # Bottom-left
            [0, 0],                # Top-left
            [court_length, 0],     # Top-right
            [court_length, court_width]  # Bottom-right
        ])

        # Convert to float32 (required by OpenCV)
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Calculate the perspective transformation matrix
        # This matrix maps any pixel coordinate → real-world coordinate
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, 
            self.target_vertices
        )

    def transform_point(self, point):
        """
        Convert a single point from pixel coordinates to real-world meters.
        
        Process:
        1. Check if point is within the calibrated field area
        2. If yes, apply perspective transformation
        3. If no, return None (point is outside trackable area)
        
        Args:
            point: Numpy array [x, y] in pixel coordinates
            
        Returns:
            Numpy array [x, y] in meters, or None if outside field area
            
        Example:
            pixel_pos = np.array([500, 600])
            field_pos = transform_point(pixel_pos)  # Returns [12.5, 34.2] in meters
        """
        # Convert to integer tuple for polygon test
        p = (int(point[0]), int(point[1]))
        
        # Check if point is inside the defined field area
        # pointPolygonTest returns >= 0 if inside, < 0 if outside
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        
        if not is_inside:
            # Point is outside the calibrated area (off-screen or in stands)
            return None

        # Reshape for OpenCV's perspective transform function
        # Required format: (1, 1, 2) instead of (2,)
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, 
            self.perspective_transformer
        )
        
        # Reshape back to simple [x, y] format
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Add real-world field positions to all tracked objects.
        
        For each object in each frame:
        - Takes the camera-adjusted pixel position
        - Converts to real-world meters on the field
        - Stores as 'position_transformed'
        
        This enables:
        - Accurate speed calculation (meters per second)
        - Distance traveled (in meters, not pixels)
        - Comparing player movements fairly (top vs bottom of screen)
        
        Args:
            tracks: Dictionary of tracking data (modified in-place)
        """
        # Process each object type (players, ball, referees)
        for object, object_tracks in tracks.items():
            # Process each frame
            for frame_num, track in enumerate(object_tracks):
                # Process each tracked object in this frame
                for track_id, track_info in track.items():
                    # Get camera-adjusted position (accounts for camera panning)
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    
                    # Transform from pixels to meters
                    position_transformed = self.transform_point(position)
                    
                    # Convert from numpy array to list for storage
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    
                    # Store transformed position
                    # Will be None if object was outside calibrated field area
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed