import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator():
    """
    Estimates camera movement between video frames using optical flow.
    
    This class tracks how the camera moves across frames by identifying 
    key points in the image and tracking how they shift. This is essential 
    for football analysis because:
    - The camera pans to follow the action
    - We need to know if a player moved, or if the camera moved
    - Allows us to calculate accurate player positions on the field
    """
    
    def __init__(self, frame):
        """
        Initialize the camera movement estimator with the first frame.
        
        Args:
            frame: The first frame of the video (BGR image)
        """
        # Minimum pixel distance to consider as actual camera movement
        # If movement is less than 5 pixels, we ignore it (could be noise/jitter)
        self.minimum_distance = 5

        # Lucas-Kanade optical flow parameters
        # Optical flow = tracking how pixels move between frames
        self.lk_params = dict(
            winSize=(15, 15),      # Size of search window (15x15 pixels)
            maxLevel=2,            # Number of pyramid levels (for multi-scale tracking)
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            # Termination criteria: stop after 10 iterations or 0.03 accuracy
        )

        # Convert first frame to grayscale (optical flow works on grayscale)
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask to tell the algorithm WHERE to look for features
        # We only want features from static parts of the image (not players/ball)
        mask_features = np.zeros_like(first_frame_grayscale)  # Start with all zeros (black)
        mask_features[:, 0:20] = 1      # Left edge of frame (likely static - sideline/ads)
        mask_features[:, 900:1050] = 1  # Right portion of frame (likely static - sideline/ads)
        # These areas are less likely to have moving players, so good for tracking camera

        # Parameters for detecting good features to track
        self.features = dict(
            maxCorners=100,        # Maximum number of corner points to detect
            qualityLevel=0.3,      # Minimum quality of corners (0-1, higher = stricter)
            minDistance=3,         # Minimum distance between detected corners
            blockSize=7,           # Size of neighborhood for corner detection
            mask=mask_features     # Only detect features in our masked areas
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust player/ball positions to account for camera movement.
        
        Problem: If camera pans right, players appear to move left in the frame
        Solution: Subtract camera movement to get "true" field positions
        
        Args:
            tracks: Dictionary containing all object tracking data
            camera_movement_per_frame: List of [x, y] camera movements for each frame
        """
        # Loop through each type of object (players, ball, referees)
        for object, object_tracks in tracks.items():
            # Loop through each frame
            for frame_num, track in enumerate(object_tracks):
                # Loop through each tracked object in this frame
                for track_id, track_info in track.items():
                    # Get the object's position in the frame
                    position = track_info['position']
                    
                    # Get how much the camera moved in this frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    
                    # Adjust position: subtract camera movement to get "field-relative" position
                    # Example: If camera moved right (+10), and player is at x=100,
                    # their actual field position is 100 - 10 = 90
                    position_adjusted = (
                        position[0] - camera_movement[0],  # Adjust X coordinate
                        position[1] - camera_movement[1]   # Adjust Y coordinate
                    )
                    
                    # Store the adjusted position back in the tracks
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None, video_name=None):
        """
        Calculate camera movement for each frame in the video.
        
        Uses optical flow to track how feature points move between frames.
        The assumption: most movement of tracked features = camera movement
        
        Args:
            frames: List of video frames
            read_from_stub: If True, load cached results instead of recalculating
            stub_path: Path to cache file (optional)
            video_name: Video filename for auto-generating cache path (optional)
            
        Returns:
            List of [x_movement, y_movement] for each frame
        """
        # Auto-generate cache file path based on video name
        if video_name is not None and stub_path is None:
            # Extract filename without extension (e.g., "match1.mp4" -> "match1")
            base_name = os.path.splitext(os.path.basename(video_name))[0]
            stub_path = f'stubs/camera_movement_{base_name}.pkl'
        
        # Try to load from cache if it exists (saves processing time)
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize: no camera movement for any frame yet
        camera_movement = [[0, 0]] * len(frames)

        # Convert first frame to grayscale for feature detection
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        # Detect good features to track in the first frame
        # These are corner points in the masked regions we defined
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Process each subsequent frame
        for frame_num in range(1, len(frames)):
            # Convert current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow: where did our tracked features move to?
            # new_features = new positions of the same points from old_features
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,           # Previous frame
                frame_gray,         # Current frame
                old_features,       # Points we're tracking from previous frame
                None,               # Output array (auto-created)
                **self.lk_params    # Optical flow parameters
            )

            # Find the feature that moved the most (this represents camera movement)
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Compare each old feature position with its new position
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()  # Flatten to [x, y]
                old_features_point = old.ravel()  # Flatten to [x, y]

                # Measure how far this feature moved
                distance = measure_distance(new_features_point, old_features_point)
                
                # Keep track of the maximum movement
                if distance > max_distance:
                    max_distance = distance
                    # Store the X and Y components of this movement
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, 
                        new_features_point
                    )
            
            # Only record movement if it's above our minimum threshold
            # This filters out tiny jitters/noise
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                
                # Re-detect features for the next iteration
                # (Features can drift or leave frame, so we periodically refresh)
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update for next iteration
            old_gray = frame_gray.copy()
        
        # Save to cache for faster future runs
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draw camera movement information on each frame for visualization.
        
        Adds a semi-transparent overlay showing X and Y camera movement values.
        Useful for debugging and understanding how much the camera is moving.
        
        Args:
            frames: List of video frames
            camera_movement_per_frame: List of [x, y] movements for each frame
            
        Returns:
            List of frames with camera movement annotations
        """
        output_frames = []

        # Process each frame
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()  # Don't modify original

            # Create semi-transparent white rectangle for text background
            overlay = frame.copy()
            cv2.rectangle(
                overlay, 
                (0, 0),           # Top-left corner
                (500, 100),       # Bottom-right corner (500 pixels wide, 100 tall)
                (255, 255, 255),  # White color
                -1                # Filled rectangle
            )
            
            # Blend the overlay with original frame (60% overlay, 40% original)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get camera movement for this frame
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            
            # Draw X movement text
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",  # Text with 2 decimal places
                (10, 30),                                 # Position
                cv2.FONT_HERSHEY_SIMPLEX,                # Font
                1,                                        # Font scale
                (0, 0, 0),                               # Black color
                3                                         # Thickness
            )
            
            # Draw Y movement text
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )

            output_frames.append(frame)

        return output_frames