"""
Football Match Analysis Pipeline

This script processes football match videos to detect and track players, analyze team 
performance, and generate annotated output videos with statistics.

Pipeline stages:
1. Object Detection & Tracking (players, ball, referees)
2. Camera Movement Correction
3. Perspective Transformation (pixel → real-world coordinates)
4. Team Assignment (jersey color clustering)
5. Ball Possession Tracking
6. Pass Detection & Analysis
7. Speed & Distance Calculation
8. Visual Annotation & Output
"""

from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from pass_detector import PassDetector


def main():
    # ========== STEP 1: Load Video ==========
    print("Loading video...")
    video_path = 'C:\\Users\\jtfar\\football_event_tracking\\input_videos\\08fd33_4.mp4'
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames")

    # ========== STEP 2: Detect and Track Objects ==========
    print("Detecting and tracking objects...")
    tracker = Tracker('models/best.pt')
    
    # Get tracking data (uses cached stub if available)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        video_name=video_path
    )
    
    # Add position coordinates to all tracked objects
    tracker.add_position_to_tracks(tracks)

    # ========== STEP 3: Estimate Camera Movement ==========
    print("Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Calculate how camera pans/moves each frame
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        video_name=video_path
    )
    
    # Adjust object positions to account for camera movement
    # This gives us "field-relative" positions instead of "screen-relative"
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ========== STEP 4: Transform Perspective ==========
    print("Transforming perspective...")
    view_transformer = ViewTransformer()
    
    # Convert pixel coordinates to real-world field coordinates (meters)
    # Accounts for camera angle and perspective distortion
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ========== STEP 5: Interpolate Ball Positions ==========
    print("Interpolating ball positions...")
    # Fill in missing ball detections (ball often occluded or missed)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"], tracks)

    # ========== STEP 6: Calculate Speed and Distance ==========
    print("Calculating player speed and distance...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Add speed (km/h) and cumulative distance (meters) for each player
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ========== STEP 7: Assign Players to Teams ==========
    print("Assigning players to teams...")
    team_assigner = TeamAssigner()

    # Find first frame with players detected
    first_frame_with_players = None
    first_frame_players = None

    for frame_num in range(min(30, len(tracks['players']))):  # Check first 30 frames
        if len(tracks['players'][frame_num]) > 0:
            first_frame_with_players = frame_num
            first_frame_players = tracks['players'][frame_num]
            print(f"Found {len(first_frame_players)} players in frame {frame_num}")
            break

    if first_frame_with_players is None:
        raise RuntimeError("No players detected in video! Check your YOLO model and video.")

    # Analyze first frame with players to determine team colors
    team_assigner.assign_team_color(
        video_frames[first_frame_with_players], 
        first_frame_players
    )
    
    # Assign each player to a team based on jersey color
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],   
                track['bbox'],
                player_id
            )
            # Store team ID and color for visualization
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # ========== STEP 8: Assign Ball Possession ==========
    print("Tracking ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # For each frame, determine which player (if any) has the ball
    for frame_num, player_track in enumerate(tracks['players']):
        # Check if ball exists in this frame
        if 1 in tracks['ball'][frame_num] and 'bbox' in tracks['ball'][frame_num][1]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        else:
            # No ball detected in this frame
            assigned_player = -1

        if assigned_player != -1:
            # Mark this player as having the ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # Record which team has possession
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # No player has ball - keep previous team's possession
            if team_ball_control:
                # Continue previous team's possession
                team_ball_control.append(team_ball_control[-1])
            else:
                # First frame and no ball - default to team 1
                team_ball_control.append(1)

    team_ball_control = np.array(team_ball_control)

    # ========== STEP 9: Detect Passes ==========
    print("Detecting passes...")
    pass_detector = PassDetector()
    pass_stats = pass_detector.detect_passes(tracks, team_ball_control)
    
    # Display pass statistics in console
    print("\n" + "="*40)
    print("         PASS STATISTICS")
    print("="*40)
    print(f"Team 1: {pass_stats['team_1']['successful']}/{pass_stats['team_1']['total']} passes "
          f"({pass_stats['team_1']['accuracy']:.1f}% accuracy)")
    print(f"Team 2: {pass_stats['team_2']['successful']}/{pass_stats['team_2']['total']} passes "
          f"({pass_stats['team_2']['accuracy']:.1f}% accuracy)")
    print("="*40 + "\n")

    # ========== STEP 10: Draw Visual Annotations ==========
    print("Drawing annotations...")
    
    # Draw player ellipses, IDs, team colors, ball markers, and statistics
    output_video_frames = tracker.draw_annotations(
        video_frames, 
        tracks, 
        team_ball_control, 
        pass_detector.passes
    )

    # Draw camera movement overlay (top-left corner)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    # Draw speed and distance for each player (below their feet)
    #speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # ========== STEP 11: Save Output Video ==========
    print("Saving output video...")
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print("✓ Analysis complete! Output saved to 'output_videos/output_video.avi'")


if __name__ == '__main__':
    main()