import cv2
import sys 
sys.path.append('../')
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator():
    """
    Calculates and displays player speed and distance traveled during the match.
    
    Uses a windowing approach: instead of calculating speed every single frame,
    we sample every N frames for efficiency and smoother results.
    """
    
    def __init__(self):
        """Initialize the estimator with frame window and frame rate settings."""
        # Calculate speed over 5-frame windows (reduces jitter from frame-to-frame noise)
        self.frame_window = 5
        
        # Video frame rate (frames per second)
        # Used to convert frame counts into real time for speed calculations
        self.frame_rate = 24
    
    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Calculate speed and cumulative distance for all players.
        
        Strategy:
        1. Sample positions every 5 frames (frame_window)
        2. Calculate distance traveled between samples
        3. Convert to speed using time elapsed
        4. Track cumulative distance for entire match
        
        Args:
            tracks: Dictionary containing all tracking data (modified in-place)
        """
        # Track total distance traveled by each player throughout the video
        total_distance = {}

        # Process each object type (players, ball, referees)
        for object, object_tracks in tracks.items():
            # Skip ball and referees - we only care about player movement
            if object == "ball" or object == "referees":
                continue 
            
            number_of_frames = len(object_tracks)
            
            # Loop through frames in windows (0, 5, 10, 15, ...)
            # This is more efficient than calculating speed for every single frame
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Calculate the last frame in this window
                # Use min() to avoid going past the end of the video
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # Process each tracked player in the starting frame
                for track_id, _ in object_tracks[frame_num].items():
                    # Skip if player isn't tracked in the ending frame
                    # (player might have left the field or tracking lost them)
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get player positions at start and end of window
                    # Using 'position_transformed' = real-world field coordinates (in meters)
                    # Not pixel coordinates - this accounts for perspective and camera angle
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Skip if transformation failed for either position
                    if start_position is None or end_position is None:
                        continue
                    
                    # Calculate distance traveled (in meters)
                    distance_covered = measure_distance(start_position, end_position)
                    
                    # Calculate time elapsed between frames (in seconds)
                    # Example: 5 frames at 24 fps = 5/24 = 0.208 seconds
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    
                    # Calculate speed: distance / time = meters per second
                    speed_meters_per_second = distance_covered / time_elapsed
                    
                    # Convert to km/h (multiply by 3.6)
                    # Why 3.6? Because 1 m/s = 3.6 km/h
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Initialize tracking structures if needed
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    # Add this window's distance to cumulative total
                    total_distance[object][track_id] += distance_covered

                    # Assign the calculated speed and distance to ALL frames in this window
                    # This creates smooth, consistent values across the window
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        
                        # Store speed and cumulative distance in the track data
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self, frames, tracks):
        """
        Draw speed and distance information on video frames.
        
        For each player, displays:
        - Current speed in km/h
        - Total distance traveled in meters
        
        Text appears below each player's feet.
        
        Args:
            frames: List of video frames
            tracks: Dictionary containing tracking data with speed/distance
            
        Returns:
            List of frames with speed/distance annotations
        """
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            # Process each object type
            for object, object_tracks in tracks.items():
                # Only draw for players (skip ball and referees)
                if object == "ball" or object == "referees":
                    continue 
                
                # Draw for each tracked player in this frame
                for _, track_info in object_tracks[frame_num].items():
                    # Only draw if speed data exists
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        if speed is None or distance is None:
                            continue
                        
                        # Get player's bounding box
                        bbox = track_info['bbox']
                        
                        # Get foot position (bottom-center of bounding box)
                        position = get_foot_position(bbox)
                        position = list(position)
                        
                        # Offset text below the player's feet (40 pixels down)
                        position[1] += 40
                        position = tuple(map(int, position))
                        
                        # Draw speed text
                        cv2.putText(
                            frame, 
                            f"{speed:.2f} km/h",        # 2 decimal places
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,                         # Font scale
                            (0, 0, 0),                   # Black color
                            2                            # Thickness
                        )
                        
                        # Draw distance text (20 pixels below speed)
                        cv2.putText(
                            frame, 
                            f"{distance:.2f} m",
                            (position[0], position[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2
                        )
            
            output_frames.append(frame)
        
        return output_frames