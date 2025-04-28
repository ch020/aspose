"""
BLAZEPOSE WRAPPER
"""
import numpy as np, cv2
from pathlib import Path

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
        if not _mp_ok:
            return []

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
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                keypoints = {
                    np.solutions.pose.PoseLandmark(idx).name: np.array([lm.x, lm.y, lm.z])
                    for idx, lm in enumerate(res.pose_landmarks.landmark)
                }
                track.append(keypoints)
            else:
                track.append({})
        cap.release()
        pose.close()
        return track