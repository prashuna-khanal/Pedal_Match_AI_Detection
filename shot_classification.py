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
            # Index 24 is RIGHT_HIP, 23 is LEFT_HIP
            right_shoulder = pose.landmark[12]
            left_shoulder = pose.landmark[11]
            right_wrist = pose.landmark[16]
            right_hip = pose.landmark[24]
            left_hip = pose.landmark[23]
            
            # Visibility threshold
            visibility_threshold = 0.2
            if (getattr(right_shoulder, 'visibility', 1.0) < visibility_threshold or 
                getattr(left_shoulder, 'visibility', 1.0) < visibility_threshold or 
                getattr(right_wrist, 'visibility', 1.0) < visibility_threshold or
                getattr(right_hip, 'visibility', 1.0) < visibility_threshold):
                continue
                
            shot_type = "None"
            
            # Rule 1: Smash - wrist is raised VERY high above the shoulder
            if right_wrist.y < right_shoulder.y - 0.08:
                shot_type = "Smash"
                
            # Rule 2: Forehand - wrist extended right AND raised above the hip (not just resting at side)
            elif right_wrist.x > right_shoulder.x + 0.04 and right_wrist.y < right_hip.y:
                shot_type = "Forehand"
                
            # Rule 3: Backhand - wrist crossed over to left shoulder AND raised above hip
            elif right_wrist.x < left_shoulder.x - 0.02 and right_wrist.y < left_hip.y:
                shot_type = "Backhand"
                
            if shot_type != "None":
                detected_shots.append((player_idx, shot_type))
                
        return detected_shots
