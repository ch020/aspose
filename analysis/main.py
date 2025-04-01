import json
import os
import zipfile
from typing import Dict, Optional, List

import cv2
import mediapipe as mp
from scipy.signal import savgol_filter

from config import *
from maths_helpers import *

mp_pose = mp.solutions.pose

def extract_all_zips(zip_dir: str = ZIP_INPUT_DIR, output_dir: str = EXTRACTION_OUTPUT_DIR) -> Dict[str, Optional[Dict]]:
    os.makedirs(output_dir, exist_ok=True)
    all_data = {}

    for filename in os.listdir(zip_dir):
        if filename.endswith(".zip"):
            zip_path = os.path.join(zip_dir, filename)
            participant_id = os.path.splitext(filename)[0]
            participant_dir = os.path.join(output_dir, participant_id)
            os.makedirs(participant_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(participant_dir)
            print(f"Extracted {filename} to {participant_dir}")

            metadata_path = os.path.join(participant_dir, "metadata.json")
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"Loaded metadata for {participant_id}: {metadata}")
            else:
                print(f"No metadata for {participant_id}")

            video_files = sorted(
                [f for f in os.listdir(participant_dir) if f.endswith(".mp4")],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            video_paths = [os.path.join(participant_dir, f) for f in video_files]

            all_data[participant_id] = {
                "metadata": metadata,
                "video_paths": video_paths
            }

    return all_data

def read_video_frames(video_path: str) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def extract_keypoints_from_frames_blazepose(frames: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    keypoints_list = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame in frames:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                keypoints = {
                    landmark.name: np.array([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])
                    for landmark in mp_pose.PoseLandmark
                }
                keypoints_list.append(keypoints)
            else:
                keypoints_list.append(None)

    return keypoints_list

def normalised_to_cm(value: float, user_height_cm: float, user_height_px:float) -> float:
    scale_factor = user_height_cm / user_height_px
    return value * scale_factor

def smooth_keypoints(keypoints_series: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    return savgol_filter(keypoints_series, window_length=window, polyorder=polyorder, axis=0)

def compute_cervical_rotation_3d(keypoints_series: List[Dict[str, np.ndarray]]) -> float:
    angles = []

    for keypoints in keypoints_series:
        if keypoints:
            nose = keypoints["nose"]
            left_shoulder = keypoints["left_shoulder"]
            right_shoulder = keypoints["right_shoulder"]

            torso_centre = (left_shoulder + right_shoulder) / 2

            head_vector = nose - torso_centre
            head_yaw = np.degrees(np.arctan2(head_vector[0], head_vector[2]))
            angles.append(head_yaw)

    angles = np.array(angles)

    if SMOOTHING:
        angles = smooth_keypoints(angles)

    max_rotation = np.abs(np.max(angles) - np.min(angles))
    return max_rotation


def lateral_spinal_flexion(keypoints_series: List[Dict[str, np.ndarray]],user_height_cm: float, user_height_px: float, side: str = "left") -> float:
    distances = []

    for keypoints in keypoints_series:
        if keypoints:
            wrist = keypoints[f'{side}_wrist'][:2]
            knee = keypoints[f'{side}_knee'][:2]

            distance_px = np.linalg.norm(wrist - knee)
            distances.append(distance_px)

    distances = np.array(distances)
    if SMOOTHING:
        distances = smooth_keypoints(distances)
    min_distance_px = np.min(distances)

    distance_cm = normalised_to_cm(min_distance_px, user_height_cm, user_height_px)
    return distance_cm

def tragus_to_wall(keypoints_series: List[Dict[str, np.ndarray]],user_height_cm: float, user_height_px: float, side: str = "left"):
    distances = []

    for keypoints in keypoints_series:
        if keypoints:
            ear = keypoints[f'{side}_ear'][:2]
            shoulder = keypoints[f'{side}_shoulder'][:2]

            distance_px = np.abs(ear[0] - shoulder[0])
            distances.append(distance_px)

    distances = np.array(distances)
    if SMOOTHING:
        distances = smooth_keypoints(distances)
    min_distance_px = np.min(distances)

    distance_cm = normalised_to_cm(min_distance_px, user_height_cm, user_height_px)
    return distance_cm

def intermalleolar_distance(keypoints_series: List[Dict[str, np.ndarray]],user_height_cm: float, user_height_px: float):
    distances = []

    for keypoints in keypoints_series:
        if keypoints:
            left_ankle = keypoints[f'left_ankle'][:2]
            right_ankle = keypoints[f'right_ankle'][:2]

            distance_px = np.abs(left_ankle - right_ankle)
            distances.append(distance_px)

    distances = np.array(distances)
    if SMOOTHING:
        distances = smooth_keypoints(distances)
    max_distance_px = np.max(distances)

    distance_cm = normalised_to_cm(max_distance_px, user_height_cm, user_height_px)
    return distance_cm

if __name__ == "__main__":
    metadata_dict = extract_all_zips()  # Extract data zips
