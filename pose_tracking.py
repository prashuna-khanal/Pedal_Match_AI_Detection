import cv2
import mediapipe as mp
import copy

class PoseTracker:
    def __init__(self):
        # 1. Initialize the standard single-person MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, 
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        Process a single frame by slicing it into 4 quadrants to detect small players.
        Returns a list of detected poses and the annotated frame.
        """
        height, width, _ = frame.shape
        
        # Calculate midpoints
        mid_y = height // 2
        mid_x = width // 2
        
        # Define the 4 quadrants: (start_y, end_y, start_x, end_x, offset_x_norm, offset_y_norm)
        # offset_x_norm is either 0.0 or 0.5 (used to remap coordinates back to the full frame)
        quadrants = [
            (0, mid_y, 0, mid_x, 0.0, 0.0),             # Top-Left
            (0, mid_y, mid_x, width, 0.5, 0.0),         # Top-Right
            (mid_y, height, 0, mid_x, 0.0, 0.5),        # Bottom-Left
            (mid_y, height, mid_x, width, 0.5, 0.5)     # Bottom-Right
        ]
        
        list_of_poses = []
        
        # MediaPipe expects RGB images
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for (y1, y2, x1, x2, offset_x, offset_y) in quadrants:
            # Crop the quadrant
            quadrant_rgb = image_rgb[y1:y2, x1:x2]
            
            # Process just this quadrant
            results = self.pose.process(quadrant_rgb)
            
            if results.pose_landmarks:
                # We found a person! But their coordinates are relative to the tiny quadrant (0.0 to 1.0)
                # We must map them back to the full image coordinates.
                remapped_landmarks = copy.deepcopy(results.pose_landmarks)
                
                for landmark in remapped_landmarks.landmark:
                    # landmark.x is 0.0 to 1.0 inside the quadrant.
                    # Since the quadrant is exactly half the width of the full image:
                    landmark.x = (landmark.x / 2.0) + offset_x
                    landmark.y = (landmark.y / 2.0) + offset_y
                    # Z is also relative to width, so we halve it to maintain scale
                    landmark.z = landmark.z / 2.0
                
                list_of_poses.append(remapped_landmarks)
                
                # Draw the adjusted landmarks onto the FULL frame
                self.mp_drawing.draw_landmarks(
                    frame, 
                    remapped_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
            
        # Draw some faint crosshairs so the user can see the quadrants
        cv2.line(frame, (mid_x, 0), (mid_x, height), (255, 255, 255), 1)
        cv2.line(frame, (0, mid_y), (width, mid_y), (255, 255, 255), 1)
            
        return list_of_poses, frame
