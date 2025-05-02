"""
BLAZEPOSE WRAPPER
"""
from pathlib import Path
import cv2
import numpy as np

try:
    import mediapipe as mp
    _mp_ok = True
except  ModuleNotFoundError:
    _mp_ok = False
    mp = None

class Backend:
    name = "blazepose"

    def __init__(self):
        if not _mp_ok:
            print("[blazepose] MediaPipe not installed - skipping")

    def infer(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        """Runs Blazepose on a video and returns a list of keypoint dictionaries."""
        if not _mp_ok:
            return []

        # Initialise the BlazePose model
        pose = mp.solutions.pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        track = []
        cap = cv2.VideoCapture(str(video_path))

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Rotate video to portrait if in landscape
            h, w, _ = frame.shape
            if h < w:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                h, w = w, h

            # Run BlazePose on the frame
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_landmarks:
                keypoints = {
                    mp.solutions.pose.PoseLandmark(idx).name.lower(): np.array([
                        lm.x * w, # convert x from normalised to pixels
                        lm.y * h, # convert y from normalised to pixels
                        lm.z * w, # approximate z scaling by width
                    ])
                    for idx, lm in enumerate(res.pose_landmarks.landmark)
                }
                track.append(keypoints)
            else:
                track.append({})
        cap.release()
        pose.close()
        return track
