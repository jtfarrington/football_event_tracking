from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns players to teams based on jersey color using K-means clustering.
    
    Strategy:
    1. Extract dominant color from each player's jersey
    2. Cluster all players into 2 groups (the two teams)
    3. Remember each player's team assignment
    """
    
    def __init__(self):
        """Initialize team assignment tracking structures."""
        # Store the representative color for each team (BGR format)
        # Example: {1: [120, 45, 200], 2: [30, 180, 50]}
        self.team_colors = {}
        
        # Cache of player ID -> team ID assignments
        # Once we know a player's team, we don't recalculate
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        """
        Create a K-means model to find the 2 dominant colors in an image.
        
        K-means will identify:
        - Cluster 0: One dominant color (e.g., jersey)
        - Cluster 1: Another dominant color (e.g., background/skin)
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            Fitted KMeans model
        """
        # Reshape image from (height, width, 3) to (num_pixels, 3)
        # Each row becomes a single pixel's BGR values
        image_2d = image.reshape(-1, 3)

        # K-means with 2 clusters (jersey color vs everything else)
        # init="k-means++": Smart initialization for faster convergence
        # n_init=1: Only run once (we don't need multiple attempts for simple case)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant jersey color for a single player.
        
        Process:
        1. Crop to player's bounding box
        2. Use only top half (jersey area, avoiding shorts/socks)
        3. Cluster into 2 colors
        4. Identify which cluster is the jersey (not background)
        
        Args:
            frame: Video frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            BGR color array representing jersey color
        """
        # Crop frame to player's bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Use only top half of crop (this is where the jersey is)
        # Bottom half has shorts, socks, grass - we don't want those colors
        top_half_image = image[0:int(image.shape[0]/2), :]

        # Cluster pixels into 2 groups: jersey vs background
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster assignment for each pixel
        labels = kmeans.labels_

        # Reshape back to image dimensions for spatial analysis
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identify background cluster using corner pixels
        # Assumption: corners are likely background/field, not player
        corner_clusters = [
            clustered_image[0, 0],      # Top-left
            clustered_image[0, -1],     # Top-right
            clustered_image[-1, 0],     # Bottom-left
            clustered_image[-1, -1]     # Bottom-right
        ]
        
        # Whichever cluster appears most in corners = background
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # The other cluster must be the player/jersey
        player_cluster = 1 - non_player_cluster

        # Get the average BGR color of the player cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Determine the representative colors for both teams.
        
        Called once at the start using the first frame.
        
        Process:
        1. Extract jersey color for each detected player
        2. Cluster all player colors into 2 groups (the two teams)
        3. Store team colors for future reference
        
        Args:
            frame: First video frame
            player_detections: Dictionary of player detections in first frame
        """

        # Safety check: need at least 2 players to assign teams
        if len(player_detections) < 2:
            raise ValueError(f"Need at least 2 players to assign teams, got {len(player_detections)}")
        
        # Extract jersey color for every player in the frame
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        # Cluster all player colors into 2 teams
        # n_init=10: Run 10 times and pick best result (more robust)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Save the model for later use in get_player_team()
        self.kmeans = kmeans

        # Store representative color for each team
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine which team a player belongs to.
        
        Uses caching: once we know a player's team, we remember it.
        
        Args:
            frame: Current video frame
            player_bbox: Player's bounding box
            player_id: Unique player tracking ID
            
        Returns:
            Team ID (1 or 2)
        """
        # Check cache first - if we already know this player's team, return it
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extract this player's jersey color
        player_color = self.get_player_color(frame, player_bbox)

        # Predict which team cluster this color belongs to
        # Returns 0 or 1, we add 1 to get team IDs 1 or 2
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        # Special case override (likely goalkeeper or specific player)
        # TODO: Make this more robust - maybe detect goalkeepers separately
        if player_id == 91:
            team_id = 1

        # Cache this assignment for future frames
        self.player_team_dict[player_id] = team_id

        return team_id