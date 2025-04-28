"""
OPENPOSE WRAPPER
"""
import numpy as np, json, subprocess, tempfile, os
from pathlib import Path

BODY_25_KEYPOINTS = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "background"
]


class Backend:
    name = "openpose"

    def __init__(self):
        try:
            import pyopenpose
            self._mode = "python"
        except ModuleNotFoundError:
            self._bin = "openpose"
            self._mode = "cli"

    def infer(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        if self._mode == "python":
            return self._infer_python(video_path)
        return self._infer_cli(video_path)

    def _infer_python(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        import cv2, pyopenpose as op
        op_wrapper = op.WrapperPython()
        op_wrapper.configure({
            "model_pose": "BODY_25",
            "disable_multi_thread": True,
        })
        op_wrapper.start()

        datum = op.Datum()
        cap = cv2.VideoCapture(str(video_path))
        track = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            datum.cvInputData = frame
            op_wrapper.emplaceAndPop([datum])
            if datum.poseKeypoints is not None and len(datum.poseKeypoints):
                pts = datum.poseKeypoints[0]
                frame_keypoints = {BODY_25_KEYPOINTS[i]: pts[i] for i in range(min(len(BODY_25_KEYPOINTS), pts.shape[0]))}
                track.append(frame_keypoints)
            else:
                track.append({})
        cap.release()
        op_wrapper.stop()
        return track

    def _infer_cli(self, video_path) -> list[dict[str, np.ndarray]]:
        with tempfile.TemporaryDirectory() as td:
            cmd = [self._bin,
                   "--video", str(video_path),
                   "--write_json", td,
                   "--display", "0",
                   "--render_pose", "0"]
            try:
                subprocess.run(cmd,
                               check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"[openpose CLI] Error: {e}")
                return []

            frames = sorted(Path(td).glob("*.json"), key=lambda p: int(p.stem.split("_")[-1]))
            track = []
            for f in frames:
                data = json.loads(f.read_text())
                if data["people"]:
                    pts = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
                    frame_keypoints = {BODY_25_KEYPOINTS[i]: pts[i] for i in range(min(len(BODY_25_KEYPOINTS), pts.shape[0]))}
                    track.append(frame_keypoints)
                else:
                    track.append({})
            return track