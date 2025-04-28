"""
HRNET WRAPPER
"""
import numpy as np, cv2, os
from pathlib import Path

try:
    from mmpose.apis import (inference_topdown, init_model)
    from mmengine import Config
    _mm_ok = True
except  ModuleNotFoundError:
    _mm_ok = False

from analysis import config as C

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class Backend:
    name = "hrnet"

    def __init__(self):
        if not _mm_ok:
            print("[hrnet] mmpose not installed - skipping")
            return
        cfg_path = Path(C.WEIGHTS_DIR, "hrnet_w32_coco_256x192.py")
        cfg = Config.fromfile(cfg_path)
        cfg.model.init_cfg = dict(
            type='Pretrained',
            checkpoint=str(Path(C.WEIGHTS_DIR, "hrnet_w32_coco_256x192.pth"))
        )
        self.det_model = init_model(cfg, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    def infer(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        if not _mm_ok:
            return []
        cap = cv2.VideoCapture(str(video_path))
        track = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w, _ = frame.shape
            bbox = [0, 0, w, h]
            res = inference_topdown(
                self.det_model,
                frame,
                [{"bbox": bbox}],
                bbox_format="xyxy"
            )
            if res and res[0]["score"] > 0.2:
                pts = res[0]["keypoints"]
                frame_keypoints = {COCO_KEYPOINTS[i]: pts[i] for i in range(min(len(COCO_KEYPOINTS), pts.shape[0]))}
                track.append(frame_keypoints)
            else:
                track.append({})
        cap.release()
        return track