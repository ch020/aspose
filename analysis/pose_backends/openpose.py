"""
OPENPOSE WRAPPER
"""
import numpy as np, json, subprocess, tempfile, os
from pathlib import Path
import os
from analysis import config as C
import re

# OpenPose BODY-25 keypoint labels
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
        # Path to OpenPose binary
        self._bin  = str(C.OPENPOSE_BIN / "OpenPoseDemo.exe")
        os.environ["PATH"] += os.pathsep + str(C.OPENPOSE_BIN)  # Ensure DLLs are discoverable

    def infer(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        """Runs OpenPose on a video file using the CLI and returns a list of keypoints per frame."""
        video_path_str = video_path.resolve().as_posix()

        with tempfile.TemporaryDirectory() as td:
            json_dir = Path(td).as_posix()
            cmd = [self._bin,
                   "--video", video_path_str,
                   "--write_json", json_dir,
                   "--display", "0",
                   "--render_pose", "0"]
            try:
                subprocess.run(cmd,
                               check=True,
                               cwd=str(C.OPENPOSE_ROOT),)
            except subprocess.CalledProcessError as e:
                print(f"[openpose CLI] Error: {e}")
                if e.stderr:
                    print(e.stderr.decode())
                if e.stdout:
                    print(e.stdout.decode())
                return []

            def extract_frame_number(path: Path) -> int:
                match = re.search(r"(\d+)(?=\.json$)", path.name)
                return int(match.group()) if match else -1

            frames = sorted(Path(td).glob("*.json"), key=extract_frame_number)
            track = []
            for f in frames:
                data = json.loads(f.read_text())
                if data["people"]:
                    pts = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
                    frame_keypoints = {
                        BODY_25_KEYPOINTS[i]: pts[i]
                        for i in range(min(len(BODY_25_KEYPOINTS), pts.shape[0]))
                    }
                    track.append(frame_keypoints)
                else:
                    track.append({})
            return track
