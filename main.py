import cv2
import pandas as pd
import os

from pose_tracking import PoseTracker
from shot_classification import ShotClassifier
from ball_tracking import BallTracker

def main():
    
    # 1.Video Path
    video_path = os.path.join('.','data','infernce_sample_video.mp4')
    if not os.path.exists(video_path):
        print(f"Error: Could not find video at {video_path}")
        return

    # 2. Initialize our simple modules
    tracker = PoseTracker()
    classifier = ShotClassifier()
    ball_tracker = BallTracker()
    
    # 3. Open the video file
    video = cv2.VideoCapture(video_path)    
    
    # Prepare to store our results
    results_data = []
    frame_count = 0
    
    # Bonus Task: Shot Counts and Cooldowns to prevent multiple counting of same swing
    shot_counts = {"Forehand": 0, "Backhand": 0, "Smash": 0}
    player_cooldowns = {0: 0, 1: 0, 2: 0, 3: 0} # 30 frame cooldown per player
    
    # Bonus Task: Bounce display timer
    bounce_text_timer = 0
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to stop early.")

    # 4. Process frame by frame
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Decrease cooldowns
        for p in player_cooldowns:
            if player_cooldowns[p] > 0:
                player_cooldowns[p] -= 1
                
        # Track pose using MediaPipe
        list_of_poses, annotated_frame = tracker.process_frame(frame)
        
        # Track the ball (now returns center, bounce boolean, and direction)
        ball_center, is_bounce, ball_direction = ball_tracker.process_frame(frame)
        
        if is_bounce:
            bounce_text_timer = 15 # show text for 15 frames
            # Also log the bounce event
            results_data.append({
                "frame": frame_count,
                "player": "None",
                "shot_type": "Bounce",
                "ball_x": ball_center[0] if ball_center else None,
                "ball_y": ball_center[1] if ball_center else None,
                "ball_direction": ball_direction
            })
            
        if ball_center:
            # Draw a circle around the ball in yellow
            cv2.circle(annotated_frame, ball_center, 5, (0, 255, 255), -1)
            cv2.circle(annotated_frame, ball_center, 8, (0, 200, 200), 2)
            
            # Show the ball direction near the ball if it's moving
            if ball_direction != "None":
                cv2.putText(annotated_frame, ball_direction, (ball_center[0] + 15, ball_center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Classify the shot based on landmarks
        detected_shots = classifier.classify_shots_for_all_players(list_of_poses)
        
        # If we detected an actual shot, save the data
        for player_idx, shot_type in detected_shots:
            # Only count the shot if this player is not on cooldown
            if player_idx not in player_cooldowns:
                player_cooldowns[player_idx] = 0
                
            if player_cooldowns[player_idx] == 0:
                shot_counts[shot_type] += 1
                player_cooldowns[player_idx] = 30 # Wait 30 frames (approx 1 sec) before counting next shot for this player
            
            results_data.append({
                "frame": frame_count,
                "player": player_idx,
                "shot_type": shot_type,
                "ball_x": ball_center[0] if ball_center else None,
                "ball_y": ball_center[1] if ball_center else None,
                "ball_direction": ball_direction
            })
            
            # Print to console for real-time feedback
            print(f"Frame {frame_count} | Player {player_idx}: Detected {shot_type}")
            
            # Draw the classification on the video frame
            cv2.putText(annotated_frame, f"P{player_idx}: {shot_type}", (50, 200 + (player_idx * 40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # --- Draw Video Overlays ---
        # Draw Scoreboard Background
        cv2.rectangle(annotated_frame, (10, 10), (300, 140), (0, 0, 0), -1)
        
        # Draw Title
        cv2.putText(annotated_frame, "SHOT ANALYTICS", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # Draw Counts
        y_pos = 70
        for shot, count in shot_counts.items():
            cv2.putText(annotated_frame, f"{shot}: {count}", (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            y_pos += 30
            
        # Draw Bounce Text if active
        if bounce_text_timer > 0:
            cv2.putText(annotated_frame, "BOUNCE!", (500, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4, cv2.LINE_AA)
            bounce_text_timer -= 1

        # Show the video with landmarks and overlays
        cv2.imshow("Padel Analysis", annotated_frame)
        
        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. Clean up OpenCV resources
    video.release()
    cv2.destroyAllWindows()
    
    # 6. Save results using Pandas
    if results_data:
        df = pd.DataFrame(results_data)
        os.makedirs("output", exist_ok=True)
        output_file = os.path.join("output", "classification_results.csv")
        df.to_csv(output_file, index=False)
        print(f"\nSuccess! Analysis complete. Results saved to {output_file}")
    else:
        print("\nAnalysis complete. No shots detected.")

if __name__ == "__main__":
    main()
