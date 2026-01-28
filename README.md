# Football Event Tracking System

A computer vision system for analyzing football (soccer) matches using YOLO object detection, ByteTrack, and advanced tracking algorithms. This project processes match videos to extract detailed statistics including player movements, team possession, pass completion rates, speed, and distance traveled.

![](https://github.com/jtfarrington/football_event_tracking/blob/main/my_gif.gif)

## Core Functionality

- Object Detection & Tracking: Detects and tracks players, referees, and the ball across video frames
- Team Assignment: Automatically assigns players to teams using jersey color clustering (K-means)
- Ball Possession Tracking: Determines which player has the ball and calculates possession percentages
- Pass Detection: Identifies successful passes, interceptions, and calculates pass accuracy 
- Speed & Distance Calculation: Measures player speed (km/h) and distance traveled (meters)
- Camera Movement Compensation: Adjusts for camera panning to get accurate field-relative positions
- Perspective Transformation: Converts pixel coordinates to real-world field measurements

### Visual Output

- Player tracking with team-colored ellipses and ID numbers
- Ball position indicator
- Possession indicator (red triangle) on player with ball

Real-time statistics overlay:

- Ball possession percentages
- Pass statistics with accuracy
- Camera movement data

### Tech Stack

- Python 3.9+
- YOLOv8 (Ultralytics) - Object detection
- ByteTrack (Supervision) - Multi-object tracking
- OpenCV - Video processing and computer vision
- NumPy & Pandas - Data processing
- scikit-learn - K-means clustering for team assignment

### Processing Pipeline

1. **Video Loading**: Read video frames into memory
2. **Object Detection**: YOLO detects players, ball, and referees
3. **Object Tracking**: ByteTrack maintains consistent IDs across frames
4. **Camera Movement**: Optical flow estimates camera panning
5. **Perspective Transform**: Convert pixel coordinates to real-world meters
6. **Team Assignment**: K-means clustering on jersey colors
7. **Ball Assignment**: Proximity-based ball possession detection
8. **Pass Detection**: Analyze possession changes for pass events ⭐
9. **Speed/Distance**: Calculate movement metrics in real-world units
10. **Annotation**: Draw all visualizations on frames
11. **Output**: Save annotated video

### Key Algorithms

**Team Assignment:**
- Extract jersey colors using K-means (2 clusters per player)
- Cluster all players into 2 teams
- Cache assignments for consistency

**Pass Detection:** 
- Track possession changes frame-by-frame
- Filter out tracking glitches (require ball to be "free" for ≥1 frame)
- Classify as successful pass (same team) or interception (different team)
- Calculate accuracy percentages

**Speed Calculation:**
- Sample positions every 5 frames (reduces noise)
- Transform to real-world coordinates
- Calculate: speed = distance / time
- Convert to km/h

## Output Statistics

The system provides:
- **Team 1 & 2 Ball Possession**: Percentage of time each team controlled the ball
- **Pass Statistics**: 
  - Total passes attempted
  - Successful passes completed
  - Pass accuracy percentage
- **Player Metrics**:
  - Current speed (km/h)
  - Total distance traveled (meters)
- **Camera Movement**: X/Y movement in pixels

## Acknowledgments

This project was built following the excellent tutorial by codeinajiffy Abdullah Tarek.(https://www.youtube.com/@codeinajiffy) The core tracking and team assignment functionality is based on their work.

## Future Enhancements

- [ ] Heatmap generation (player positioning)
- [ ] Shot detection and analysis
- [ ] Offside detection
- [ ] Formation analysis
- [ ] Pass network visualization
- [ ] Real-time processing support
- [ ] Web dashboard for statistics
- [ ] Player jersey number recognition
- [ ] Multi-camera support
- [ ] Export statistics to CSV/JSON

## Known Issues

- Ball detection can be inconsistent when occluded (mitigated by interpolation)
- Goalkeeper detection sometimes requires manual override (hardcoded fix for player #91)
- Perspective transformation is video-specific (requires manual calibration for new camera angles)
- First run is slow (~5-30 minutes depending on video length and hardware)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
