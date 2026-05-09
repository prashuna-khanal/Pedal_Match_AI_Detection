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
    
    print(f"Processing video: {video_path}")
    print("Press 'q' to stop early.")

    # 4. Process frame by frame
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Track pose using MediaPipe
        list_of_poses, annotated_frame = tracker.process_frame(frame)
        
        # Track the ball
        ball_center = ball_tracker.process_frame(frame)
        if ball_center:
            # Draw a circle around the ball in yellow
            cv2.circle(annotated_frame, ball_center, 5, (0, 255, 255), -1)
            cv2.circle(annotated_frame, ball_center, 8, (0, 200, 200), 2)
        
        # Classify the shot based on landmarks
        detected_shots = classifier.classify_shots_for_all_players(list_of_poses)
        
        # If we detected an actual shot, save the data
        for player_idx, shot_type in detected_shots:
            results_data.append({
                "frame": frame_count,
                "player": player_idx,
                "shot_type": shot_type,
                "ball_x": ball_center[0] if ball_center else None,
                "ball_y": ball_center[1] if ball_center else None
            })
            
            # Print to console for real-time feedback
            print(f"Frame {frame_count} | Player {player_idx}: Detected {shot_type} (Ball at {ball_center})")
            
            # Draw the classification on the video frame
            cv2.putText(annotated_frame, f"P{player_idx}: {shot_type}", (50, 50 + (player_idx * 40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the video with landmarks
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
