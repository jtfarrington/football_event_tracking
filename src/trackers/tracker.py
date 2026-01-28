from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from ball_tracker import UltimateBallTracker


class Tracker:
    """
    Tracks players, referees, and ball across video frames using YOLO detection and ByteTrack.
    
    Responsibilities:
    - Detect objects in each frame using YOLO model
    - Track objects across frames (maintains consistent IDs)
    - Draw visual annotations on frames
    - Cache results to avoid re-processing
    """
    
    def __init__(self, model_path):
        """
        Initialize the tracker with a trained YOLO model.
        
        Args:
            model_path: Path to the YOLO weights file (best.pt)
        """
        # Load YOLO model for object detection
        self.model = YOLO(model_path) 
        
        # ByteTrack: Maintains consistent IDs for objects across frames
        # Even if detection is lost for a few frames, ByteTrack remembers the object
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """
        Calculate and add position coordinates for all tracked objects.
        
        Position meaning:
        - Ball: Center of bounding box (ball is small, center is fine)
        - Players/Referees: Foot position (bottom-center of box - where they stand on field)
        
        Args:
            tracks: Dictionary of tracking data (modified in-place)
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    
                    # Different position logic for ball vs players
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        # For players/referees, use foot position (where they touch the ground)
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions, tracks=None):
        """
        Ultimate ball tracking with all advanced techniques.
        
        Features:
        - Multi-hypothesis tracking
        - Confidence scoring
        - Size validation
        - Boundary constraints (if configured)
        - Temporal smoothing
        """
        
        # Optional: Define field boundaries for constraint
        # Format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        # These should be the corners of the visible field area
        # Set to None if you don't have field calibration
        field_boundaries = None  # Or define your field corners
        
        # Create tracker
        tracker = UltimateBallTracker(field_boundaries=field_boundaries)
        
        # Run tracking
        return tracker.interpolate_ball_positions(ball_positions, tracks)

    def detect_frames(self, frames):
        """
        Run YOLO detection on all video frames in batches.
        
        Batching improves GPU efficiency - processing 20 frames at once is faster
        than processing 20 frames individually.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of YOLO detection results
        """
        batch_size = 20 
        detections = [] 
        
        # Process frames in batches of 20
        for i in range(0, len(frames), batch_size):
            # conf=0.1: Low confidence threshold (10%)
            # We keep more detections to avoid missing the ball/players
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            
        return detections
    
    def tighten_bbox(self, bbox, shrink_factor=1):
        """
        Shrink a bounding box by a percentage to make it tighter.
        
        Args:
            bbox: [x1, y1, x2, y2]
            shrink_factor: How much to shrink (0.1 = 10% smaller)
        
        Returns:
            Tightened bbox [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        
        # Shrink from all sides
        shrink_x = width * shrink_factor / 2
        shrink_y = height * shrink_factor / 2
        
        x1 += shrink_x
        x2 -= shrink_x
        y1 += shrink_y
        y2 -= shrink_y
        
        return [x1, y1, x2, y2]
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, video_name=None):
        """
        Detect and track all objects (players, referees, ball) across all frames.
        """
        # Auto-generate cache filename based on video name
        if video_name is not None and stub_path is None:
            base_name = os.path.splitext(os.path.basename(video_name))[0]
            stub_path = f'stubs/track_stubs_{base_name}.pkl'
        
        # Try to load from cache first
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # No cache - run fresh detection
        detections = self.detect_frames(frames)

        # Initialize tracking structure
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Process each frame's detections
        for frame_num, detection in enumerate(detections):
            # Get class names from YOLO model
            cls_names = detection.names
            # Create reverse mapping: name -> class_id
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert YOLO format to Supervision format (for ByteTrack)
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Treat goalkeepers as regular players (same team assignment logic)
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # ByteTrack: Assign consistent IDs across frames
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize this frame's tracking dictionaries
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Store tracked objects ← THIS LOOP MUST BE INDENTED INSIDE THE FRAME LOOP!
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    # Tighten player bounding boxes by 10%
                    tight_bbox = self.tighten_bbox(bbox, shrink_factor=0.1)
                    tracks["players"][frame_num][track_id] = {"bbox": tight_bbox}
                
                if cls_id == cls_names_inv['referee']:
                    # Tighten referee boxes too
                    tight_bbox = self.tighten_bbox(bbox, shrink_factor=0.1)
                    tracks["referees"][frame_num][track_id] = {"bbox": tight_bbox}
            
            # Store ball detections separately (not tracked with ByteTrack)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save to cache for next time
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse at player's feet with optional ID number.
        
        Visual style:
        - Ellipse shows where player is standing on field
        - Small rectangle contains player's tracking ID number
        
        Args:
            frame: Video frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: BGR color tuple
            track_id: Player ID number (optional)
            
        Returns:
            Modified frame
        """
        # Get bottom-center of bounding box (player's feet)
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse at feet
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),  # Oval shape
            angle=0.0,
            startAngle=0,
            endAngle=360,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw ID number in small rectangle
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            # Draw filled rectangle background
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            
            # Adjust text position for 3-digit IDs
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            # Draw ID number
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        """
        Draw a triangle marker (used for ball and possession indicator).
        
        Args:
            frame: Video frame
            bbox: Bounding box
            color: BGR color
            
        Returns:
            Modified frame
        """
        # Position triangle at top of bounding box
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Define triangle points (pointing up)
        triangle_points = np.array([
            [x, y],          # Bottom point (center)
            [x - 10, y - 20],  # Top-left
            [x + 10, y - 20],  # Top-right
        ])
        
        # Draw filled triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # Draw black outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw ball possession statistics in bottom-right corner.
        
        Shows cumulative possession percentage for each team up to current frame.
        
        Args:
            frame: Video frame
            frame_num: Current frame number
            team_ball_control: Array of team IDs (1 or 2) for each frame
            
        Returns:
            Modified frame
        """
        # Draw semi-transparent white rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate possession up to current frame only
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        
        # Count frames each team had possession
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        
        # Convert to percentages
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Draw possession percentages
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame
    
    def draw_pass_statistics(self, frame, frame_num, all_passes):
        """
        Draw pass statistics in bottom-left corner.
        
        Shows cumulative pass counts and accuracy for each team up to current frame.
        Updates dynamically as passes occur during the video.
        
        Args:
            frame: Video frame
            frame_num: Current frame number
            all_passes: List of all pass events (filtered by frame number here)
            
        Returns:
            Modified frame
        """
        # Only count passes that have occurred up to this frame
        passes_so_far = [p for p in all_passes if p['frame'] <= frame_num]
        
        # Separate by team
        team_1_passes = [p for p in passes_so_far if p['from_team'] == 1]
        team_2_passes = [p for p in passes_so_far if p['from_team'] == 2]
        
        # Count successful passes
        team_1_successful = len([p for p in team_1_passes if p['successful']])
        team_2_successful = len([p for p in team_2_passes if p['successful']])
        
        # Total passes
        team_1_total = len(team_1_passes)
        team_2_total = len(team_2_passes)
        
        # Calculate accuracy percentages
        team_1_accuracy = (team_1_successful / team_1_total * 100) if team_1_total > 0 else 0
        team_2_accuracy = (team_2_successful / team_2_total * 100) if team_2_total > 0 else 0
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 850), (500, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw pass statistics text
        team_1_text = f"Team 1 Passes: {team_1_successful}/{team_1_total} ({team_1_accuracy:.1f}%)"
        cv2.putText(frame, team_1_text, (40, 900), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        team_2_text = f"Team 2 Passes: {team_2_successful}/{team_2_total} ({team_2_accuracy:.1f}%)"
        cv2.putText(frame, team_2_text, (40, 940), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, pass_stats=None):
        """
        Draw all visual annotations on video frames.
        
        Draws:
        - Player ellipses with team colors and ID numbers
        - Referee ellipses (yellow)
        - Ball triangle marker (green)
        - Possession indicator (red triangle) on player with ball
        - Ball control statistics (bottom-right)
        - Pass statistics (bottom-left)
        
        Args:
            video_frames: List of video frames
            tracks: Dictionary of tracking data
            team_ball_control: Array of ball possession per frame
            pass_stats: List of pass events (optional)
            
        Returns:
            List of annotated frames
        """
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Get tracking data for this frame
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw all players
            for track_id, player in player_dict.items():
                # Use team color if available, default to blue
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # Draw possession indicator if this player has the ball
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw all referees (yellow)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            # Draw ball with confidence-based colors ← UPDATE THIS SECTION
            for track_id, ball in ball_dict.items():
                # Check if this is a real detection or just interpolated
                is_real = ball.get("is_real_detection", True)
                confidence = ball.get("confidence", 1.0)
                
                if is_real:
                    # Real detection - draw in green
                    frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))
                elif confidence >= 0.5:
                    # High-confidence prediction - draw in yellow
                    frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 255))
                # Low confidence predictions (<0.5) are not drawn at all

            # Draw statistics overlays
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            if pass_stats is not None:
                frame = self.draw_pass_statistics(frame, frame_num, pass_stats)

            output_video_frames.append(frame)

        return output_video_frames