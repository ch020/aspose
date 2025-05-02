"""
HRNET WRAPPER
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from analysis import config as C
from ..hrnet.lib.config import cfg, update_config
from ..hrnet.lib.models.pose_hrnet import get_pose_net
from ..hrnet.lib.utils.transforms import get_affine_transform, transform_preds

# Define the COCO keypoint order
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class Backend:
    name = "hrnet"

    def __init__(self):
        # Load model configurations and weights
        config_path = Path(C.HRNET_ROOT / "experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml")
        model_path = Path(C.HRNET_ROOT / "pose_hrnet_w32_256x192.pth")

        # Create config arguments
        args = argparse.Namespace(
            cfg=str(config_path),
            opts=[],
            modelDir='',
            logDir='',
            dataDir='',
            prevModelDir=''
        )

        # Load and apply configuration
        update_config(cfg, args)
        self.model = get_pose_net(cfg, is_train=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval().cuda()

        # Image input size from configuration
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.aspect_ratio = self.image_size[0] / self.image_size[1]

    def infer(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        cap = cv2.VideoCapture(str(video_path))
        track = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w, _ = frame.shape

            centre = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            scale = np.array([w, h], dtype=np.float32) / 200.0
            trans = get_affine_transform(centre, scale, 0, self.image_size)

            # Apply affine transformation and prepare tensor
            input_img = cv2.warpAffine(frame, trans, tuple(self.image_size.astype(int)), flags=cv2.INTER_LINEAR)
            input_img = input_img.astype(np.float32) / 255.0
            input_img = input_img.transpose(2, 0, 1)
            input_tensor = torch.from_numpy(input_img).unsqueeze(0).cuda()

            # Perform inference
            with torch.no_grad():
                output = self.model(input_tensor)
                heatmap = output.cpu().numpy()[0]

            # Extract keypoints from heatmaps and transform to original image space
            preds = []
            for i in range(heatmap.shape[0]):
                idx = np.unravel_index(np.argmax(heatmap[i]), heatmap[i].shape)
                y, x = idx[0], idx[1]
                pred = transform_preds(np.array([[x, y]]), centre, scale, self.image_size)
                preds.append(pred[0])
                print(pred[0])

            # Store keypoints in a dictionary using the COCO keypoint names
            frame_keypoints = {COCO_KEYPOINTS[i]: preds[i]
                               for i in range(len(COCO_KEYPOINTS))}
            track.append(frame_keypoints)

        cap.release()
        return track
