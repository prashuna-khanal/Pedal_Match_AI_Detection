# Simple Padel Shot Classifier

A minimal prototype for analyzing Padel gameplay footage using OpenCV and MediaPipe. 

This project was built to be as simple and readable as possible, focusing on core computer vision concepts without the overhead of complex deep learning pipelines like YOLO or PyTorch.

## Features

1. **Player Tracking**: Uses MediaPipe Pose to extract body landmarks (shoulders, elbows, wrists).
2. **Ball Tracking**: Uses simple OpenCV color thresholding (HSV) to find and track the yellow padel ball.
3. **Shot Classification**: Uses basic coordinate rules to guess the shot type:
   - **Forehand**: Right wrist extended to the right.
   - **Backhand**: Right wrist crossed to the left.
   - **Smash**: Right wrist raised high above the shoulder.
4. **Structured Output**: Saves all detected shots and ball coordinates to `output/classification_results.csv` using Pandas.

## Project Structure

- `main.py`: The main orchestrator. Opens the video, runs trackers, and saves data.
- `pose_tracking.py`: A simple wrapper around MediaPipe to extract body landmarks.
- `ball_tracking.py`: Uses HSV color ranges to find the padel ball.
- `shot_classification.py`: Contains the simple if/else rules to determine the shot type based on the pose.

## Installation

1. Clone or download the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Place a video file in the `data/` folder and name it `infernce_sample_video.mp4` (or update the path in `main.py`).

Then, run the script:
```bash
python main.py
```

The script will open a preview window showing the tracking in real-time. Press `q` to stop early.
Once completed, check the `output/` folder for your `classification_results.csv`.

# Output Video
https://drive.google.com/file/d/1Ck7WReqRgmSus-s1Nlpvfeom05gJnq_e/view?usp=sharing
