"""
Smart Ball Tracker - Hybrid approach using multiple tracking techniques.

Combines:
- YOLO detections (when available)
- Kalman filtering (short gaps)
- Physics-based prediction (long passes)
- Player proximity estimation (ball possession)
"""

import numpy as np
from filterpy.kalman import KalmanFilter
import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class SmartBallTracker:
    """
    Intelligent ball tracking using multiple techniques.
    
    Strategy:
    1. Use YOLO detections when available (high confidence)
    2. Use Kalman filter for short gaps (1-5 frames)
    3. Use physics model for longer gaps (5-20 frames)
    4. Estimate from player positions if ball likely possessed
    """
    
    def __init__(self):
        """Initialize the smart ball tracker."""
        # Kalman filter for ball tracking
        # State: [x, y, vx, vy] (position + velocity)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we measure position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10
        
        # Process noise
        self.kf.Q *= 0.1
        
        # Initial covariance
        self.kf.P *= 100
        
        # Tracking state
        self.last_good_detection = None  # Last [x, y] center position
        self.last_good_bbox = None       # Last bbox
        self.frames_without_detection = 0
        self.initialized = False
        
    def interpolate_ball_positions(self, ball_positions):
        """
        Main method to interpolate ball positions across all frames.
        
        Args:
            ball_positions: List of ball tracking data per frame
                           Format: [{1: {'bbox': [x1,y1,x2,y2]}}, ...]
            
        Returns:
            Smoothed/interpolated ball positions
        """
        smoothed_positions = []
        
        for frame_num, frame_data in enumerate(ball_positions):
            # Extract YOLO detection if available
            ball_bbox = frame_data.get(1, {}).get('bbox', None)
            
            # Get player context (not available in this simple version)
            # In full implementation, you'd pass player positions here
            players = {}
            
            # Track the ball
            estimated_bbox = self.track(ball_bbox, frame_num, players)
            
            # Store result
            if estimated_bbox is not None:
                smoothed_positions.append({1: {"bbox": estimated_bbox}})
            else:
                # No estimate - use last known or empty
                if self.last_good_bbox is not None:
                    smoothed_positions.append({1: {"bbox": self.last_good_bbox}})
                else:
                    smoothed_positions.append({})
        
        return smoothed_positions
        
    def track(self, yolo_bbox, frame_num, players):
        """
        Track ball using best available method.
        
        Args:
            yolo_bbox: Ball bbox [x1, y1, x2, y2] from YOLO or None
            frame_num: Current frame number
            players: Dictionary of player data {player_id: {'bbox': [...]}}
            
        Returns:
            Ball bbox [x1, y1, x2, y2] or None
        """
        if yolo_bbox is not None and len(yolo_bbox) == 4:
            # Good detection from YOLO - use it!
            return self._handle_detection(yolo_bbox)
        else:
            # No detection - need to estimate
            return self._handle_missing_detection(players)
    
    def _handle_detection(self, bbox):
        """Process a valid YOLO detection."""
        self.frames_without_detection = 0
        
        # Get center position
        center = get_center_of_bbox(bbox)
        self.last_good_detection = center
        self.last_good_bbox = bbox
        
        # Update Kalman filter
        if not self.initialized:
            # Initialize filter with first detection
            self.kf.x = np.array([center[0], center[1], 0, 0])
            self.initialized = True
        else:
            # Predict then update
            self.kf.predict()
            self.kf.update(np.array(center))
        
        return bbox
    
    def _handle_missing_detection(self, players):
        """Handle missing detection using various strategies."""
        self.frames_without_detection += 1
        
        if not self.initialized:
            # Can't predict without initialization
            return None
        
        if self.frames_without_detection <= 5:
            # Short gap - use Kalman prediction
            return self._predict_with_kalman()
        
        elif self.frames_without_detection <= 20:
            # Medium gap - use context-aware prediction
            if self._is_ball_in_flight():
                # Ball in flight - use physics
                return self._predict_with_physics()
            else:
                # Ball likely with player
                return self._estimate_from_players(players)
        
        else:
            # Long gap - give up
            return None
    
    def _predict_with_kalman(self):
        """Use Kalman filter to predict ball position."""
        # Predict next state
        self.kf.predict()
        
        # Get predicted position
        predicted_center = self.kf.x[:2]
        
        # Convert to bbox
        return self._center_to_bbox(predicted_center)
    
    def _predict_with_physics(self):
        """
        Use physics model (projectile motion) for prediction.
        
        Applies gravity to simulate ball arc during passes.
        """
        # Predict with Kalman
        self.kf.predict()
        
        # Get predicted position and velocity
        predicted_pos = self.kf.x[:2].copy()
        velocity = self.kf.x[2:4].copy()
        
        # Apply gravity (rough approximation)
        # Gravity accelerates ball downward
        gravity = 0.3 * self.frames_without_detection  # Accumulated gravity
        predicted_pos[1] += gravity  # Y increases downward in image coordinates
        
        return self._center_to_bbox(predicted_pos)
    
    def _estimate_from_players(self, players):
        """
        Estimate ball position based on closest player.
        
        Assumption: If ball not visible and not moving fast,
        it's probably at a player's feet.
        """
        if self.last_good_detection is None or not players:
            # Fall back to Kalman
            return self._predict_with_kalman()
        
        # Find closest player to last known ball position
        min_distance = float('inf')
        closest_player_bbox = None
        
        for player_id, player_data in players.items():
            player_bbox = player_data.get('bbox')
            if player_bbox is None or len(player_bbox) != 4:
                continue
            
            # Player's foot position (bottom center)
            player_foot = [
                (player_bbox[0] + player_bbox[2]) / 2,
                player_bbox[3]
            ]
            
            # Distance to last known ball position
            distance = measure_distance(player_foot, self.last_good_detection)
            
            if distance < min_distance:
                min_distance = distance
                closest_player_bbox = player_bbox
        
        # If player is close enough, assume ball is at their feet
        if closest_player_bbox is not None and min_distance < 100:
            ball_center = [
                (closest_player_bbox[0] + closest_player_bbox[2]) / 2,
                closest_player_bbox[3]  # At player's feet
            ]
            return self._center_to_bbox(ball_center)
        else:
            # No close player - use Kalman
            return self._predict_with_kalman()
    
    def _is_ball_in_flight(self):
        """
        Check if ball is likely in the air.
        
        Heuristic: If velocity is high, ball is probably in flight.
        """
        if not self.initialized:
            return False
        
        # Get velocity magnitude from Kalman state
        velocity = self.kf.x[2:4]
        speed = np.linalg.norm(velocity)
        
        # If moving fast, assume it's in flight
        # Threshold: ~20 pixels per frame
        return speed > 20
    
    def _center_to_bbox(self, center):
        """
        Convert center position to bounding box.
        
        Args:
            center: [x, y] position
            
        Returns:
            [x1, y1, x2, y2] bbox
        """
        # Assume ball is ~15 pixels in diameter
        ball_size = 15
        
        return [
            center[0] - ball_size/2,
            center[1] - ball_size/2,
            center[0] + ball_size/2,
            center[1] + ball_size/2
        ]