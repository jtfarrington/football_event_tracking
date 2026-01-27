"""
Player Ball Assignment Module

Determines which player (if any) has possession of the ball based on proximity.
Uses distance calculation between ball position and player feet positions.
"""

import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner():
    """
    Assigns ball possession to the closest player within range.
    
    Logic:
    - Calculates distance from ball to each player's feet
    - Assigns ball to closest player if within threshold distance
    - Returns -1 if no player is close enough (ball is "free")
    
    The threshold distance prevents assigning the ball when it's clearly
    in the air or too far from any player.
    """
    
    def __init__(self):
        """
        Initialize the ball assigner with distance threshold.
        
        Max distance is set to 70 pixels - this was empirically determined
        to work well for typical football videos. Adjust if needed for
        different camera angles or resolutions.
        """
        # Maximum pixel distance between ball and player to consider possession
        # If ball is farther than this from all players, it's considered "free"
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Determine which player (if any) has possession of the ball.
        
        Strategy:
        - Check distance from ball to each player's feet (left and right)
        - Use the closer foot for each player
        - Assign to nearest player if within threshold
        
        Why check both feet?
        - Player bounding boxes include entire body
        - Ball might be at left or right foot
        - Using closer foot gives more accurate possession detection
        
        Args:
            players: Dictionary of player tracking data for current frame
                    Format: {player_id: {'bbox': [x1, y1, x2, y2], ...}, ...}
            ball_bbox: Ball bounding box [x1, y1, x2, y2]
            
        Returns:
            player_id: ID of player with ball possession
            -1: No player has possession (ball is free/in air)
            
        Example:
            players = {5: {'bbox': [100, 200, 150, 300]}, 7: {'bbox': [400, 250, 450, 350]}}
            ball_bbox = [120, 280, 135, 295]
            assigned = assign_ball_to_player(players, ball_bbox)  # Returns 5
        """
        # Get ball's center position
        ball_position = get_center_of_bbox(ball_bbox)

        # Track the closest player found so far
        minimum_distance = 99999  # Start with very large number
        assigned_player = -1       # -1 means no player assigned yet

        # Check each player to find who's closest to the ball
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Calculate distance from ball to BOTH of the player's feet
            # We use the bottom corners of the bounding box as foot positions
            
            # Distance to left foot (bottom-left corner of bbox)
            distance_left = measure_distance(
                (player_bbox[0], player_bbox[-1]),  # Left foot: (x1, y2)
                ball_position
            )
            
            # Distance to right foot (bottom-right corner of bbox)
            distance_right = measure_distance(
                (player_bbox[2], player_bbox[-1]),  # Right foot: (x2, y2)
                ball_position
            )
            
            # Use whichever foot is closer to the ball
            distance = min(distance_left, distance_right)

            # Only consider this player if ball is within maximum distance threshold
            if distance < self.max_player_ball_distance:
                # If this player is closer than previous closest, update assignment
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        # Return the closest player's ID, or -1 if no player was close enough
        return assigned_player