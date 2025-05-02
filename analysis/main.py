import csv
import importlib
import json
import zipfile
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Protocol, Any

import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import zscore

import config as C
from maths_helpers import bucket, _B_TRAGUS, _B_LSF, _B_IMD, _B_CR, yaw

X_COORDINATE = 0
Y_COORDINATE = 1
Z_COORDINATE = 2

# IO Helpers
def extract_all() -> Dict[str, Dict]:
    """Extracts all participant data from zip files."""
    recs: Dict[str, Dict] = {}
    C.EXTRACTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for zf in C.ZIP_INPUT_DIR.glob("*.zip"):
        try:
            pid = zf.stem
            dest = C.EXTRACTED_OUTPUT_DIR / pid
            if not dest.exists():
                with zipfile.ZipFile(zf) as z:
                    z.extractall(dest)
            meta = json.loads((dest / "metadata.json").read_text())
            vids = sorted(dest.glob("*.mp4"), key=lambda p: int(p.stem))
            recs[pid] = {"meta": meta, "vids": vids}
        except Exception as e:
            print(f"[Warning] Skipped {zf.name} due to error: {e}")
    return recs

# Backend Loader
class PoseBackend(Protocol):
    name: str
    def infer(self, video_path: Path) -> List[Dict[str, np.ndarray]]: ...

def load_backend(tag: str) -> PoseBackend:
    """Dynamically loads a pose estimation backend."""
    return importlib.import_module(f"analysis.pose_backends.{tag}").Backend()

# Loads all backends used across all metrics
BACKENDS: Dict[str, PoseBackend] = {b: load_backend(b) for b in C.POSE_BACKENDS_TO_RUN}

# Cleaning Functions
def smooth(arr: np.ndarray) -> np.ndarray:
    """Applies Savitzky-Golay smoothing to an array (if the conditions are met)."""
    arr = arr[~np.isnan(arr)]
    if C.SMOOTHING and len(arr) >= C.SG_WINDOW_LENGTH:
        return savgol_filter(arr, C.SG_WINDOW_LENGTH, C.SG_POLY_ORDER, axis=0)
    return arr

def no_outliers(arr: np.ndarray) -> np.ndarray:
    """Removes statistical outliers from the array."""
    if arr.size < 3:
        return arr
    mask = np.abs(zscore(arr, nan_policy="omit")) < C.OUTLIER_ZSCORE
    return arr[mask]

# Metric Conversion
def px2cm(px: float, h_px: float, h_cm: float) -> float:
    """Converts a pixel measurement to centimetres."""
    return px * (h_cm / h_px)

# Metric Calculations
def _compute_shoulder_ear_cm(track: List[Dict[str, np.ndarray]],
                             h_px: float,
                             h_cm: float,
                             side: str = "left") -> float:
    """Calculates shoulder-to-ear vertical distance in centimetres."""
    dists_px = [abs(frame[f"{side}_ear"][Y_COORDINATE] - frame[f"{side}_shoulder"][Y_COORDINATE])
                for frame in track if frame and f"{side}_ear" in frame and f"{side}_shoulder" in frame]
    if not dists_px:
        return np.nan
    return px2cm(np.mean(dists_px), h_px, h_cm)

def _ttw(track: List[Dict[str, np.ndarray]],
         shoulder_ear_cm: float,
         side: str = "left") -> float:
    """Calculates tragus-to-wall distance."""
    tragus_dists = []
    shoulder_ear_dists_px = []

    for frame in track:
        if frame and f"{side}_ear" in frame and f"{side}_shoulder" in frame:
            shoulder_ear_px = abs(frame[f"{side}_ear"][Y_COORDINATE] - frame[f"{side}_shoulder"][Y_COORDINATE])
            tragus_z_diff = abs(frame[f"{side}_ear"][Z_COORDINATE] - frame[f"{side}_shoulder"][Z_COORDINATE])
            shoulder_ear_dists_px.append(shoulder_ear_px)
            tragus_dists.append(tragus_z_diff)

    if not tragus_dists or not shoulder_ear_dists_px:
        return np.nan

    tragus_z_diff_px = np.min(no_outliers(np.asarray(tragus_dists)))
    shoulder_ear_px_side = np.mean(shoulder_ear_dists_px)

    return px2cm(tragus_z_diff_px, shoulder_ear_px_side, shoulder_ear_cm)

def _cr(track: List[Dict[str, np.ndarray]]) -> float:
    """Calculates cervical rotation (yaw)."""
    yaws = [yaw((frame["left_shoulder"] + frame["right_shoulder"]) / 2, frame["nose"])
            for frame in track if frame and "nose" in frame and "left_shoulder" in frame and "right_shoulder" in frame]
    if not yaws:
        return np.nan
    arr = smooth(np.asarray(yaws))
    return arr.max()

def _lsf(track: List[Dict[str, np.ndarray]],
         h_px: float,
         h_cm: float,
         side: str = "left") -> float:
    """Calculates lumbar side flexion distance."""
    d = [
        abs(frame[f"{side}_wrist"][Y_COORDINATE] - frame[f"{side}_ankle"][Y_COORDINATE])
        for frame in track if frame and f"{side}_wrist" in frame and f"{side}_ankle" in frame
    ]
    if not d:
        return np.nan
    arr = no_outliers(np.asarray(d))
    return px2cm(arr.max() - arr.min(), h_px, h_cm)

def _imd(track: list[dict[str, np.ndarray]],
         h_px: float,
         h_cm: float) -> float:
    """Calculates intermalleolar distance."""
    d = [
        abs(frame.get("left_ankle", np.array([np.nan, np.nan, np.nan]))[X_COORDINATE] -
            frame.get("right_ankle", np.array([np.nan, np.nan, np.nan]))[X_COORDINATE])
        for frame in track if frame
    ]
    if not d:
        return np.nan
    d_arr = no_outliers(np.asarray(d))
    if d_arr.size == 0:
        return np.nan
    return px2cm(np.max(d_arr), h_px, h_cm)

def _standing_height_px(track: List[Dict[str, np.ndarray]]) -> float:
    """Estimates standing height in pixels from pose track."""
    spans = [
        abs(frame["nose"][Y_COORDINATE] - frame.get("mid_ankle", (frame.get("left_ankle") + frame.get("right_ankle")) / 2)[Y_COORDINATE])
        for frame in track if frame and "nose" in frame and ("mid_ankle" in frame or ("left_ankle" in frame and "right_ankle" in frame))
    ]
    if not spans:
        return np.nan

    return float(np.max(spans))

def safe_average(a: float, b: float) -> float:
    """Safely average two values, ignoring NaNs."""
    valid = [v for v in [a, b] if not np.isnan(v)]
    return mean(valid) if valid else np.nan

# MAIN PROCESSING
def process_participant(pid: str, rec: Dict[str, Any], tag: str) -> List[str]:
    """Processes a single participant's videos for one backend."""
    videos: list[Path] = rec["vids"]
    h_cm = float(rec["meta"]["height"])

    backend = BACKENDS[tag]

    results = []
    total_frames = 0
    start_time = time.perf_counter()
    for video in videos:
        track = backend.infer(video)
        total_frames += len(track)
        results.append(track)
    end_time = time.perf_counter()
    avg_time_per_frame = (end_time - start_time) / total_frames if total_frames > 0 else np.nan


    def gv(ix: int) -> List[Dict[str, np.ndarray]]:
        return results[ix]

    h_px_lsfL = _standing_height_px(gv(0))
    h_px_lsfR = _standing_height_px(gv(1))
    h_px_imdL = _standing_height_px(gv(2))
    h_px_imdR = _standing_height_px(gv(3))

    shoulder_ear_cm_left = _compute_shoulder_ear_cm(gv(2), h_px_imdL, h_cm, "left")
    shoulder_ear_cm_right = _compute_shoulder_ear_cm(gv(2), h_px_imdL, h_cm, "right")

    metrics = {
            "cr_L": _cr(gv(4)) if tag in C.POSE_BACKENDS.get("cr", []) else np.nan,
            "cr_R": _cr(gv(5)) if tag in C.POSE_BACKENDS.get("cr", []) else np.nan,
            "imd_L": _imd(gv(2), h_px_imdL, h_cm) if tag in C.POSE_BACKENDS.get("imd", []) else np.nan,
            "imd_R": _imd(gv(3), h_px_imdR, h_cm) if tag in C.POSE_BACKENDS.get("imd", []) else np.nan,
            "lsf_L": _lsf(gv(0), h_px_lsfL, h_cm, "left") if tag in C.POSE_BACKENDS.get("lsf", []) else np.nan,
            "lsf_R": _lsf(gv(1), h_px_lsfR, h_cm, "right") if tag in C.POSE_BACKENDS.get("lsf", []) else np.nan,
            "ttw_L": _ttw(gv(6), shoulder_ear_cm_left, "left") if tag in C.POSE_BACKENDS.get("ttw", []) else np.nan,
            "ttw_R": _ttw(gv(7), shoulder_ear_cm_right, "right") if tag in C.POSE_BACKENDS.get("ttw", []) else np.nan,
        }

    return [pid, tag] + [v if not np.isnan(v) else -1 for v in [
            metrics["cr_L"], metrics["cr_R"],
            metrics["imd_L"], metrics["imd_R"],
            metrics["lsf_L"], metrics["lsf_R"],
            metrics["ttw_L"], metrics["ttw_R"],
            bucket(safe_average(metrics["cr_L"], metrics["cr_R"]), *_B_CR),
            bucket(safe_average(metrics["imd_L"], metrics["imd_R"]), *_B_IMD),
            bucket(safe_average(metrics["lsf_L"], metrics["lsf_R"]), *_B_LSF),
            bucket(safe_average(metrics["ttw_L"], metrics["ttw_R"]), *_B_TRAGUS),
            avg_time_per_frame
        ]]

def main() -> None:
    participants = extract_all()
    C.RESULTS_CSV.parent.mkdir(exist_ok=True)

    header = [
        "participant", "backend",
        "cr_left", "cr_right",
        "imd_left", "imd_right",
        "lsf_left", "lsf_right",
        "ttw_left", "ttw_right",
        "cr_score", "imd_score", "lsf_score", "ttw_score",
        "avg_runtime_per_frame_sec"
    ]

    output_path = C.OUTPUT_CSV
    already_done = set()

    if output_path.exists():
        with output_path.open() as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    already_done.add((row[0], row[1]))

    with output_path.open("a", newline="") as f:
        writer = csv.writer(f)

        if not output_path.stat().st_size:
            writer.writerow(header)

        for pid, rec in tqdm(participants.items(), desc="Processing Participants"):
            for tag in BACKENDS:
                if (pid, tag) in already_done:
                    print(f"[Info] Found previous {pid} data, skipping.")
                    continue
                try:
                    row = process_participant(pid, rec, tag)
                    print(f"[Info] Writing row for {pid} with backend {tag}:")
                    print(row)
                    writer.writerow(row)
                except Exception as e:
                    print(f"[Warning] Skipped {pid} with backend {tag} due to error: {e}")

if __name__ == "__main__":
    main()
