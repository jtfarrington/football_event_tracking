import numpy as np


class PassDetector:
    """
    Detects and analyzes pass events in football match videos.
    
    A pass is defined as:
    1. Player A has possession of the ball
    2. Ball becomes "free" (no player has it) for at least 1 frame
    3. Player B (different player) gains possession
    4. If A and B are on the same team = successful pass
    5. If A and B are on different teams = interception/turnover
    
    This class tracks all pass events and calculates statistics like:
    - Total passes per team
    - Successful passes per team
    - Pass accuracy percentage per team
    """
    
    def __init__(self):
        """
        Initialize the PassDetector.
        
        Creates an empty list to store all detected pass events throughout the video.
        Each pass event will contain information about the passer, receiver, teams, and success.
        """
        self.passes = []  # List to store all pass event dictionaries
        
    def detect_passes(self, tracks, team_ball_control):
        """
        Analyze the video frame-by-frame to detect when passes occur.
        
        Logic:
        - Loop through every frame
        - Track which player has the ball
        - When possession changes from Player A to Player B:
            * Check if ball was "free" between them (indicates a pass, not a tracking glitch)
            * Check if they're on the same team (successful) or different teams (turnover)
            * Record the pass event
        
        Args:
            tracks: Dictionary containing all object tracking data
                    Format: tracks['players'][frame_num][player_id]['has_ball']
            team_ball_control: Array showing which team (1 or 2) controlled ball each frame
                              (Not directly used but passed for potential future enhancements)
            
        Returns:
            Dictionary with complete pass statistics for both teams
        """
        # Initialize tracking variables for the previous frame
        previous_player = None  # ID of player who had ball in previous frame
        previous_team = None    # Team (1 or 2) that had ball in previous frame
        frames_without_ball = 0 # Counter for frames where ball is "free" (no player has it)
        
        # Loop through each frame in the video
        for frame_num, player_track in enumerate(tracks['players']):
            # Reset current frame variables
            current_player = None  # Will store ID of player with ball in THIS frame
            current_team = None    # Will store team of player with ball in THIS frame
            
            # Search through all players in this frame to find who has the ball
            for player_id, track in player_track.items():
                # Check if this player has the 'has_ball' flag set to True
                if track.get('has_ball', False):
                    current_player = player_id
                    current_team = track['team']
                    break  # Found the player with ball, stop searching
            
            # Case 1: No player has the ball in this frame (ball is "free" / in the air)
            if current_player is None:
                frames_without_ball += 1  # Increment counter
                continue  # Skip to next frame
            
            # Case 2: A player has the ball - check if possession changed from previous frame
            if previous_player is not None and current_player != previous_player:
                # Possession DID change! But is it a real pass or just a tracking error?
                
                # Filter: Only count as pass if ball was free for at least 1 frame
                # Why? This eliminates tracking glitches where the system briefly
                # "loses" who has the ball, causing false possession changes
                if frames_without_ball >= 1:
                    # This is a legitimate pass! Create a pass event record
                    pass_event = {
                        'frame': frame_num,              # When did the pass complete?
                        'from_player': previous_player,  # Who passed the ball?
                        'to_player': current_player,     # Who received the ball?
                        'from_team': previous_team,      # Which team passed?
                        'to_team': current_team,         # Which team received?
                        'successful': previous_team == current_team  # Same team = success!
                    }
                    
                    # Add this pass event to our running list of all passes
                    self.passes.append(pass_event)
            
            # Update tracking variables for next frame's iteration
            previous_player = current_player
            previous_team = current_team
            frames_without_ball = 0  # Reset counter (player has ball now)
        
        # After analyzing all frames, calculate and return statistics
        return self.get_pass_statistics()
    
    def get_pass_statistics(self):
        """
        Calculate comprehensive pass statistics from all detected pass events.
        
        For each team, calculates:
        - Total passes attempted (including successful and unsuccessful)
        - Successful passes (to a teammate)
        - Pass accuracy percentage (successful / total * 100)
        
        Note: An "unsuccessful" pass means the other team intercepted it or won possession
        
        Returns:
            Dictionary containing statistics for both teams:
            {
                'team_1': {
                    'total': int,           # Total passes attempted by team 1
                    'successful': int,      # Successful passes by team 1
                    'accuracy': float       # Percentage (0-100)
                },
                'team_2': { ... }  # Same structure for team 2
            }
        """
        # Edge case: No passes detected at all (maybe very short video or tracking issues)
        if len(self.passes) == 0:
            return {
                'team_1': {'total': 0, 'successful': 0, 'accuracy': 0},
                'team_2': {'total': 0, 'successful': 0, 'accuracy': 0}
            }
        
        # Filter passes by which team initiated them
        # Team 1 passes = all passes where 'from_team' is 1
        team_1_passes = [p for p in self.passes if p['from_team'] == 1]
        # Team 2 passes = all passes where 'from_team' is 2
        team_2_passes = [p for p in self.passes if p['from_team'] == 2]
        
        # Count successful passes (where 'successful' flag is True)
        # Successful = pass went to a teammate, not intercepted
        team_1_successful = len([p for p in team_1_passes if p['successful']])
        team_2_successful = len([p for p in team_2_passes if p['successful']])
        
        # Count total pass attempts
        team_1_total = len(team_1_passes)
        team_2_total = len(team_2_passes)
        
        # Build the statistics dictionary
        stats = {
            'team_1': {
                'total': team_1_total,
                'successful': team_1_successful,
                # Calculate accuracy: avoid division by zero if no passes attempted
                'accuracy': (team_1_successful / team_1_total * 100) if team_1_total > 0 else 0
            },
            'team_2': {
                'total': team_2_total,
                'successful': team_2_successful,
                'accuracy': (team_2_successful / team_2_total * 100) if team_2_total > 0 else 0
            }
        }
        
        return stats