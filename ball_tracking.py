import cv2
import numpy as np

class BallTracker:
    def __init__(self):
        # Define the color range for a yellow/green Padel ball in HSV space
        # These values might need slight tweaking depending on the video's lighting
        self.lower_color = np.array([25, 40, 40])
        self.upper_color = np.array([65, 255, 255])
        
    def process_frame(self, frame):
        """
        Process a single frame to find the ball based on color.
        Returns the (x, y) coordinates of the ball center, or None if not found.
        """
        # Convert the frame to HSV (Hue, Saturation, Value) color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask that isolates the yellow/green colors
        mask = cv2.inRange(hsv_frame, self.lower_color, self.upper_color)
        
        # Clean up the mask with some simple morphological operations (remove small noise)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours (outlines of the shapes) in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_center = None
        
        if len(contours) > 0:
            # Find the largest contour (most likely the ball if it's the only yellow thing)
            # You can also filter by area or circularity here if needed
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Make sure it's not just a tiny blip of noise
            if cv2.contourArea(largest_contour) > 10:
                # Get the minimum enclosing circle to find the center
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                
                # We also only care if the radius is reasonable for a ball (not huge)
                if radius < 50:
                    ball_center = (int(x), int(y))
                    
        return ball_center
