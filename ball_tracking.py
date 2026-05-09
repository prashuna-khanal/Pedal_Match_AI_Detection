import cv2
import numpy as np
from collections import deque

class BallTracker:
    def __init__(self):
        # Define the color range for a yellow/green Padel ball in HSV space
        self.lower_color = np.array([25, 40, 40])
        self.upper_color = np.array([65, 255, 255])
        
        # Keep track of the last few positions to determine direction and bounce
        self.history = deque(maxlen=5)
        self.last_bounce_frame = -100
        self.frame_count = 0
        
    def process_frame(self, frame):
        """
        Process a single frame to find the ball based on color.
        Returns the (x, y) coordinates of the ball center, 
        plus rule-based 'bounce' (boolean) and 'direction' ("Left", "Right", "None").
        """
        self.frame_count += 1
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.lower_color, self.upper_color)
        
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_center = None
        is_bounce = False
        direction = "None"
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 10:
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                if radius < 50:
                    ball_center = (int(x), int(y))
                    self.history.appendleft(ball_center)
        else:
            self.history.clear()
            
        # --- Simple Rule-Based Logic ---
        if len(self.history) >= 3:
            current_y = self.history[0][1]
            prev_y = self.history[1][1]
            older_y = self.history[2][1]
            
            # Direction Logic (X-axis)
            current_x = self.history[0][0]
            older_x = self.history[2][0]
            if current_x > older_x + 5:
                direction = "Right"
            elif current_x < older_x - 5:
                direction = "Left"
                
            # Bounce Logic (Y-axis changes from increasing to decreasing)
            # In OpenCV, Y increases as you go DOWN the screen.
            # So falling = Y is increasing
            # Bouncing = Y was increasing, but now Y is decreasing
            
            if (prev_y > older_y) and (current_y < prev_y):
                # Only register a bounce if we haven't seen one in the last 10 frames (cooldown)
                if self.frame_count - self.last_bounce_frame > 10:
                    is_bounce = True
                    self.last_bounce_frame = self.frame_count
                    
        return ball_center, is_bounce, direction
