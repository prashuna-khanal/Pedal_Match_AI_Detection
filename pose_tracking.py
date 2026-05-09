import cv2
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class PoseTracker:
    def __init__(self, num_poses=4):
        # 1. Download model if it doesn't exist (needed for multi-pose tracking)
        model_path = 'pose_landmarker_lite.task'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe multi-pose model...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
            urllib.request.urlretrieve(url, model_path)

        # 2. Initialize MediaPipe PoseLandmarker to support multiple people
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=num_poses, # This enables multi-person tracking!
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def process_frame(self, frame):
        """
        Process a single frame and return a list of detected poses and the annotated frame.
        """
        # Convert OpenCV BGR format to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert RGB image to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process the image to find multiple poses
        detection_result = self.detector.detect(mp_image)
        
        # Draw all the detected poses on the original frame
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                # We need to convert the landmarks to the format expected by drawing_utils
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    self.mp_pose.POSE_CONNECTIONS
                )
            
        return detection_result.pose_landmarks, frame
