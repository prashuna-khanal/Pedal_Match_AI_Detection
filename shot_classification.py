import mediapipe as mp

class ShotClassifier:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
    def classify_shots_for_all_players(self, list_of_poses):
        """
        Takes a list of poses (each pose is a list of landmarks) and applies simple rules.
        Returns a list of tuples: [(player_index, shot_type), ...]
        """
        detected_shots = []
        if not list_of_poses:
            return detected_shots
            
        for player_idx, pose in enumerate(list_of_poses):
            # Index 12 is RIGHT_SHOULDER, 11 is LEFT_SHOULDER, 16 is RIGHT_WRIST
            right_shoulder = pose[12]
            left_shoulder = pose[11]
            right_wrist = pose[16]
            
            # Visibility threshold
            visibility_threshold = 0.5
            # MediaPipe tasks API provides a visibility score for each landmark
            if (getattr(right_shoulder, 'visibility', 1.0) < visibility_threshold or 
                getattr(left_shoulder, 'visibility', 1.0) < visibility_threshold or 
                getattr(right_wrist, 'visibility', 1.0) < visibility_threshold):
                continue
                
            shot_type = "None"
            
            # Rule 1: Smash - if wrist is significantly higher than the shoulder
            # (Y value is smaller, since 0 is top)
            if right_wrist.y < right_shoulder.y - 0.1:
                shot_type = "Smash"
                
            # Assuming the player is facing AWAY from the camera (typical broadcast angle):
            # Right shoulder has a larger X value than left shoulder.
            # Rule 2: Forehand - wrist is extended to the right of the right shoulder
            elif right_wrist.x > right_shoulder.x + 0.05:
                shot_type = "Forehand"
                
            # Rule 3: Backhand - wrist is brought across the body, to the left of the left shoulder
            elif right_wrist.x < left_shoulder.x - 0.05:
                shot_type = "Backhand"
                
            if shot_type != "None":
                detected_shots.append((player_idx, shot_type))
                
        return detected_shots
