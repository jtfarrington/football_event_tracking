"""
Ultimate Ball Tracker - Production-grade ball tracking system.

Combines multiple advanced techniques:
- Multi-hypothesis tracking
- Confidence scoring
- Size validation
- Field boundary constraints
- Temporal smoothing
"""

import numpy as np
from filterpy.kalman import KalmanFilter
import cv2
import os
import sys

# Ensure utils can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import get_center_of_bbox, measure_distance


class BallHypothesis:
    """
    Single hypothesis for ball trajectory.
    
    Each hypothesis maintains:
    - Kalman filter for position/velocity/size
    - Confidence score
    - Age (frames since created)
    """
    
    def __init__(self, initial_position, initial_size=15):
        """
        Initialize a ball tracking hypothesis.
        
        Args:
            initial_position: [x, y] starting position
            initial_size: Initial ball diameter in pixels
        """
        # Kalman filter with 5D state: [x, y, vx, vy, size]
        self.kf = KalmanFilter(dim_x=5, dim_z=3)
        
        # State transition matrix
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0,  0],  # x = x + vx*dt
            [0, 1, 0,  dt, 0],  # y = y + vy*dt
            [0, 0, 1,  0,  0],  # vx = vx
            [0, 0, 0,  1,  0],  # vy = vy
            [0, 0, 0,  0,  1]   # size = size
        ])
        
        # Measurement matrix [x, y, size]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ])
        
        # Noise parameters
        self.kf.R *= 10  # Measurement noise
        self.kf.Q *= 0.1  # Process noise
        self.kf.P *= 100  # Initial covariance
        
        # Initialize state
        self.kf.x = np.array([
            initial_position[0],
            initial_position[1],
            0,  # vx
            0,  # vy
            initial_size
        ])
        
        # Hypothesis metadata
        self.confidence = 1.0
        self.age = 0
        self.last_detection_age = 0
        
    def predict(self):
        """Predict next state."""
        self.kf.predict()
        self.age += 1
        self.last_detection_age += 1
        
    def update(self, measurement):
        """
        Update with measurement [x, y, size].
        
        Args:
            measurement: [x, y, size] or None
        """
        if measurement is not None:
            self.kf.update(measurement)
            self.last_detection_age = 0
        
    def get_position(self):
        """Get current position estimate."""
        return self.kf.x[:2].copy()
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.kf.x[2:4].copy()
    
    def get_size(self):
        """Get current size estimate."""
        return self.kf.x[4]
    
    def get_bbox(self):
        """Get bounding box [x1, y1, x2, y2]."""
        pos = self.get_position()
        size = self.get_size()
        return [
            pos[0] - size/2,
            pos[1] - size/2,
            pos[0] + size/2,
            pos[1] + size/2
        ]


class UltimateBallTracker:
    """
    Production-grade ball tracker with multiple advanced techniques.
    
    Features:
    - Multi-hypothesis tracking (maintains 3 possible trajectories)
    - Confidence scoring (0-1 for each prediction)
    - Size validation (rejects impossible ball sizes)
    - Field boundary constraints (keeps ball in-bounds)
    - Temporal smoothing (reduces jitter)
    """
    
    def __init__(self, field_boundaries=None):
        """
        Initialize the ultimate ball tracker.
        
        Args:
            field_boundaries: List of [x,y] points defining field boundary
                            (optional, enables boundary constraints)
        """
        # Multiple hypotheses
        self.hypotheses = []
        self.max_hypotheses = 3
        
        # Temporal smoothing buffer
        self.position_buffer = []
        self.smoothing_window = 5
        
        # Field boundaries (optional)
        if field_boundaries is not None:
            self.field_polygon = np.array(field_boundaries, dtype=np.float32)
        else:
            self.field_polygon = None
        
        # Size constraints
        self.min_ball_size = 8   # Minimum reasonable ball size
        self.max_ball_size = 40  # Maximum reasonable ball size
        
        # Confidence thresholds
        self.min_confidence_to_show = 0.3  # Don't show if confidence < 30%
        
        # State tracking
        self.last_real_detection = None
        self.frames_since_detection = 0
        
    def interpolate_ball_positions(self, ball_positions, tracks=None):
        """
        Main entry point for ball tracking across all frames.
        
        Args:
            ball_positions: List of ball detections per frame
            tracks: Full tracking data (for player context)
            
        Returns:
            Smoothed ball positions with confidence flags
        """
        smoothed_positions = []
        
        for frame_num, frame_data in enumerate(ball_positions):
            # Extract YOLO detection
            ball_bbox = frame_data.get(1, {}).get('bbox', None)
            
            # Get player context
            players = {}
            if tracks is not None and frame_num < len(tracks['players']):
                players = tracks['players'][frame_num]
            
            # Track the ball
            result = self.track_frame(ball_bbox, players)
            
            # Store result
            smoothed_positions.append(result)
        
        return smoothed_positions
    
    def track_frame(self, yolo_bbox, players):
        """
        Track ball in a single frame.
        
        Args:
            yolo_bbox: YOLO detection [x1,y1,x2,y2] or None
            players: Dict of player data for this frame
            
        Returns:
            Dict with ball data and metadata
        """
        # Step 1: Validate detection (if present)
        valid_detection = None
        if yolo_bbox is not None:
            valid_detection = self._validate_detection(yolo_bbox)
        
        # Step 2: Update hypotheses
        if valid_detection is not None:
            self._handle_detection(valid_detection, players)
            self.frames_since_detection = 0
        else:
            self._handle_no_detection(players)
            self.frames_since_detection += 1
        
        # Step 3: Select best hypothesis
        best_hypothesis = self._select_best_hypothesis()
        
        if best_hypothesis is None:
            return {}
        
        # Step 4: Apply boundary constraints
        position = best_hypothesis.get_position()
        if self.field_polygon is not None:
            position = self._constrain_to_field(position)
        
        # Step 5: Apply temporal smoothing
        smoothed_position = self._smooth_position(position)
        
        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(
            best_hypothesis, 
            valid_detection is not None,
            players
        )
        
        # Update confidence in hypothesis
        best_hypothesis.confidence = confidence
        
        # Step 7: Create output
        if confidence >= self.min_confidence_to_show:
            size = best_hypothesis.get_size()
            bbox = [
                smoothed_position[0] - size/2,
                smoothed_position[1] - size/2,
                smoothed_position[0] + size/2,
                smoothed_position[1] + size/2
            ]
            
            return {
                1: {
                    "bbox": bbox,
                    "is_real_detection": valid_detection is not None,
                    "confidence": confidence
                }
            }
        else:
            # Low confidence - don't show ball
            return {}
    
    def _validate_detection(self, bbox):
        """
        Validate that a detection is reasonable for a ball.
        
        Checks:
        - Size is in reasonable range
        - Size hasn't changed dramatically from previous
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            [x, y, size] measurement or None if invalid
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        size = (width + height) / 2
        
        # Check absolute size
        if size < self.min_ball_size or size > self.max_ball_size:
            return None
        
        # Check size consistency with previous detections
        if self.hypotheses:
            best = self._select_best_hypothesis()
            if best is not None:
                previous_size = best.get_size()
                size_change_ratio = abs(size - previous_size) / previous_size
                
                if size_change_ratio > 0.6:  # 60% change is suspicious
                    return None
        
        # Valid detection
        center = get_center_of_bbox(bbox)
        return [center[0], center[1], size]
    
    def _handle_detection(self, measurement, players):
        """
        Handle a valid ball detection.
        
        Updates existing hypotheses and creates new one.
        """
        # Update all existing hypotheses
        for hyp in self.hypotheses:
            hyp.predict()
            hyp.update(measurement)
        
        # Create new hypothesis from this detection
        new_hyp = BallHypothesis(measurement[:2], measurement[2])
        self.hypotheses.append(new_hyp)
        
        # Prune hypotheses
        self._prune_hypotheses()
        
        # Remember this detection
        self.last_real_detection = measurement
    
    def _handle_no_detection(self, players):
        """Handle frame with no ball detection."""
        # Just predict all hypotheses
        for hyp in self.hypotheses:
            hyp.predict()
            
            # Decay confidence over time
            time_penalty = 0.95 ** hyp.last_detection_age
            hyp.confidence *= time_penalty
        
        # Prune low-confidence hypotheses
        self._prune_hypotheses()
    
    def _select_best_hypothesis(self):
        """
        Select the most confident hypothesis.
        
        Returns:
            Best BallHypothesis or None
        """
        if not self.hypotheses:
            return None
        
        # Sort by confidence
        best = max(self.hypotheses, key=lambda h: h.confidence)
        return best
    
    def _prune_hypotheses(self):
        """
        Remove unlikely hypotheses.
        
        Pruning criteria:
        - Keep only top N by confidence
        - Remove very old hypotheses
        - Remove very low confidence
        """
        # Remove very low confidence
        self.hypotheses = [h for h in self.hypotheses if h.confidence > 0.1]
        
        # Remove very old hypotheses
        self.hypotheses = [h for h in self.hypotheses if h.age < 60]
        
        # Keep only top N
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        self.hypotheses = self.hypotheses[:self.max_hypotheses]
    
    def _calculate_confidence(self, hypothesis, has_detection, players):
        """
        Calculate confidence score for current ball position.
        
        Factors:
        - Time since last detection
        - Velocity smoothness
        - Proximity to players
        - Detection presence
        
        Returns:
            Confidence score 0-1
        """
        confidence = 1.0
        
        # Factor 1: Time decay
        time_penalty = 0.92 ** hypothesis.last_detection_age
        confidence *= time_penalty
        
        # Factor 2: Detection boost
        if has_detection:
            confidence = min(confidence * 1.5, 1.0)
        
        # Factor 3: Velocity smoothness
        velocity = hypothesis.get_velocity()
        speed = np.linalg.norm(velocity)
        
        # Penalize very high speeds (likely error)
        if speed > 80:
            confidence *= 0.5
        
        # Factor 4: Player proximity boost
        if players:
            position = hypothesis.get_position()
            min_dist = self._distance_to_nearest_player(position, players)
            
            if min_dist < 50:  # Close to player
                confidence *= 1.2
            elif min_dist > 200:  # Far from any player
                confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _distance_to_nearest_player(self, ball_position, players):
        """Calculate distance to nearest player."""
        min_dist = float('inf')
        
        for player_data in players.values():
            player_bbox = player_data.get('bbox')
            if player_bbox is None:
                continue
            
            player_foot = [
                (player_bbox[0] + player_bbox[2]) / 2,
                player_bbox[3]
            ]
            
            dist = measure_distance(ball_position, player_foot)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _constrain_to_field(self, position):
        """
        Constrain ball position to field boundaries.
        
        If outside field, snap to nearest point on boundary.
        """
        if self.field_polygon is None:
            return position
        
        # Check if inside field
        is_inside = cv2.pointPolygonTest(
            self.field_polygon,
            tuple(position),
            False
        ) >= 0
        
        if is_inside:
            return position
        
        # Outside field - find nearest boundary point
        return self._snap_to_boundary(position)
    
    def _snap_to_boundary(self, point):
        """Find nearest point on field boundary."""
        min_dist = float('inf')
        nearest_point = point
        
        # Check each edge of polygon
        n = len(self.field_polygon)
        for i in range(n):
            p1 = self.field_polygon[i]
            p2 = self.field_polygon[(i + 1) % n]
            
            # Project point onto line segment
            proj = self._project_point_to_segment(point, p1, p2)
            dist = np.linalg.norm(point - proj)
            
            if dist < min_dist:
                min_dist = dist
                nearest_point = proj
        
        return nearest_point
    
    def _project_point_to_segment(self, point, seg_start, seg_end):
        """Project point onto line segment."""
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            return seg_start
        
        t = np.dot(point_vec, seg_vec) / seg_len_sq
        t = np.clip(t, 0, 1)
        
        return seg_start + t * seg_vec
    
    def _smooth_position(self, position):
        """
        Apply temporal smoothing to reduce jitter.
        
        Uses exponentially weighted moving average.
        """
        self.position_buffer.append(position)
        
        # Keep buffer at window size
        if len(self.position_buffer) > self.smoothing_window:
            self.position_buffer.pop(0)
        
        # Weighted average (recent frames weighted more)
        weights = np.exp(np.linspace(-1, 0, len(self.position_buffer)))
        weights /= weights.sum()
        
        smoothed = np.average(
            self.position_buffer,
            axis=0,
            weights=weights
        )
        
        return smoothed