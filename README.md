# Football Match Analysis System

A comprehensive computer vision system for analyzing football (soccer) matches using YOLO object detection, ByteTrack, and advanced tracking algorithms. This project processes match videos to extract detailed statistics including player movements, team possession, pass completion rates, speed, and distance traveled.

## Features
### Core Functionality

Object Detection & Tracking: Detects and tracks players, referees, and the ball across video frames

Team Assignment: Automatically assigns players to teams using jersey color clustering (K-means)

Ball Possession Tracking: Determines which player has the ball and calculates possession percentages

Pass Detection: Identifies successful passes, interceptions, and calculates pass accuracy 

Speed & Distance Calculation: Measures player speed (km/h) and distance traveled (meters)

Camera Movement Compensation: Adjusts for camera panning to get accurate field-relative positions

Perspective Transformation: Converts pixel coordinates to real-world field measurements

Visual Output

Player tracking with team-colored ellipses and ID numbers

Ball position indicator

Possession indicator (red triangle) on player with ball

Real-time statistics overlay:

- Ball possession percentages
- Pass statistics with accuracy
- Camera movement data


Demo
Input:

Raw football match video

Output:

Annotated video with:

Player tracking and team identification
Ball tracking with interpolation for missing frames
Live statistics (possession, passes, speed, distance)
Camera movement overlay

Tech Stack

Python 3.9+
YOLOv8 (Ultralytics) - Object detection
ByteTrack (Supervision) - Multi-object tracking
OpenCV - Video processing and computer vision
NumPy & Pandas - Data processing
scikit-learn - K-means clustering for team assignment

Prerequisites

Python 3.9 or higher
NVIDIA GPU with CUDA 12.1+ (recommended for faster processing)
8GB+ RAM
Trained YOLO model weights (best.pt)
