import csv
import importlib
import json
import zipfile
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Protocol

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import zscore

import config as C
from maths_helpers import bucket, _B_TRAGUS, _B_LSF, _B_IMD, _B_CR, yaw


# IO Helpers
def extract_all() -> Dict[str, Dict]:
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
    return importlib.import_module(f"analysis.pose_backends.{tag}").Backend()

BACKENDS: dict[str, PoseBackend] = {}
for tag in {b for lst in C.POSE_BACKENDS.values() for b in lst}:
    BACKENDS[tag] = load_backend(tag)

# Cleaning
def smooth(arr: np.ndarray) -> np.ndarray:
    arr = arr[~np.isnan(arr)]
    if C.SMOOTHING and len(arr) >= C.SG_WINDOW_LENGTH:
        return savgol_filter(
            arr,
            C.SG_WINDOW_LENGTH,
            C.SG_POLY_ORDER,
            axis=0
        )
    return arr

def no_outliers(arr: np.ndarray) -> np.ndarray:
    if arr.size < 3:
        return arr
    mask = np.abs(zscore(arr, nan_policy="omit")) < C.OUTLIER_ZSCORE
    return arr[mask]

# Metrics
def px2cm(px: float, h_px: float, h_cm: float) -> float:
    return px * (h_cm / h_px)


def _compute_shoulder_ear_cm(track: list[dict[str, np.ndarray]],
                             h_px: float,
                             h_cm: float,
                             side: str = "left") -> float:
    dists_px = []
    for frame in track:
        if frame and f"{side}_ear" in frame and f"{side}_shoulder" in frame:
            dists_px.append(abs(frame[f"{side}_ear"][1] - frame[f"{side}_shoulder"][1]))
    if not dists_px:
        return np.nan
    avg_dist_px = np.mean(dists_px)
    shoulder_ear_cm = px2cm(avg_dist_px, h_px, h_cm)
    return shoulder_ear_cm

def _ttw(track: list[dict[str, np.ndarray]],
         shoulder_ear_cm: float,
         side: str = "left") -> float:
    tragus_dists = []
    shoulder_ear_dists_px = []

    for frame in track:
        if frame and f"{side}_ear" in frame and f"{side}_shoulder" in frame:
            shoulder_ear_px = abs(frame[f"{side}_ear"][1] - frame[f"{side}_shoulder"][1])
            tragus_z_diff = abs(frame[f"{side}_ear"][2] - frame[f"{side}_shoulder"][2])
            shoulder_ear_dists_px.append(shoulder_ear_px)
            tragus_dists.append(tragus_z_diff)

    if not tragus_dists or not shoulder_ear_dists_px:
        return np.nan

    tragus_z_diff_px = np.min(no_outliers(np.asarray(tragus_dists)))
    shoulder_ear_px_side = np.mean(shoulder_ear_dists_px)

    return px2cm(tragus_z_diff_px, shoulder_ear_px_side, shoulder_ear_cm)

def _cr(track: list[dict[str, np.ndarray]]) -> float:
    yaws = []
    for k in track:
        if k and "nose" in k and "left_shoulder" in k and "right_shoulder" in k:
            nose = k["nose"]
            torso = (k["left_shoulder"] + k["right_shoulder"]) / 2
            yaws.append(yaw(torso, nose))
    if not yaws:
        return np.nan
    arr = smooth(np.asarray(yaws))
    return arr.max() - arr.min()

def _lsf(track: list[dict[str, np.ndarray]],
         h_px: float,
         h_cm: float,
         side: str = "left") -> float:
    d = [
        abs(k[f"{side}_wrist"][1] - k[f"{side}_ankle"][1])
        for k in track if k and f"{side}_wrist" in k and f"{side}_ankle" in k
    ]
    if not d:
        return np.nan
    arr = no_outliers(np.asarray(d))
    return px2cm(arr.max() - arr.min(), h_px, h_cm)

def _imd(track: list[dict[str, np.ndarray]],
         h_px: float,
         h_cm: float) -> float:
    d = [
        abs(k.get("left_ankle", np.array([np.nan, np.nan, np.nan]))[0] -
            k.get("right_ankle", np.array([np.nan, np.nan, np.nan]))[0])
        for k in track if k
    ]
    if not d:
        return np.nan
    return px2cm(max(no_outliers(np.asarray(d))), h_px, h_cm)

# PROCESSING
def _standing_height_px(track: list[dict[str, np.ndarray]]) -> float:
    spans = [
        abs(k["nose"][1] - k.get("mid_ankle", (k.get("left_ankle") + k.get("right_ankle")) / 2)[1])
        for k in track if k and "nose" in k and ("mid_ankle" in k or ("left_ankle" in k and "right_ankle" in k))
    ]
    if not spans:
        return np.nan
    return float(max(spans))

def _metric_for(video: Path, backend: PoseBackend) -> list[dict[str, np.ndarray]]:
    return backend.infer(video)

def _process_attempts(video_pair: list[Path],
                      func,
                      h_px: float, h_cm: float,
                      **kw) -> tuple[float, float]:
    t1 = func(_metric_for(video_pair[0], kw["be"]), h_px, h_cm, kw.get("side", "left"))
    t2 = func(_metric_for(video_pair[1], kw["be"]), h_px, h_cm, kw.get("side", "right"))
    return t1, t2

def safe_average(a: float, b: float) -> float:
    valid = [v for v in [a, b] if not np.isnan(v)]
    if not valid:
        return np.nan
    return mean(valid)

def process_participant(pid: str, rec: dict) -> list:
    videos: list[Path] = rec["videos"]

    rows = []
    for metric, backend_list in C.POSE_BACKENDS.items():
        be = BACKENDS[backend_list[0]]
        if metric == "cr":
            stand_track = be.infer(videos[0])
        else:
            stand_track = BACKENDS[next(iter(BACKENDS))].infer(videos[0])

        h_px = _standing_height_px(stand_track)
        h_cm = rec["meta"].get("height", 170.0)

        shoulder_ear_cm_left = _compute_shoulder_ear_cm(stand_track, h_px, h_cm, "left")
        shoulder_ear_cm_right = _compute_shoulder_ear_cm(stand_track, h_px, h_cm, "right")

        inf = {}
        times = []

        for b in backend_list:
            backend = BACKENDS[b]
            for v in videos:
                start = time.perf_counter()
                result = backend.infer(v)
                end = time.perf_counter()
                times.append(end - start)
                inf[v] = result

        avg_runtime = np.mean(times) if times else np.nan

        def gv(ix): return inf[videos[ix]]

        cr_L = _cr(gv(2))
        cr_R = _cr(gv(3))

        imd_L = _imd(gv(4), h_px, h_cm)
        imd_R = _imd(gv(5), h_px, h_cm)

        lsf_L = _lsf(gv(0), h_px, h_cm, "left")
        lsf_R = _lsf(gv(1), h_px, h_cm, "right")

        ttw_L = _ttw(gv(6), shoulder_ear_cm_left, "left")
        ttw_R = _ttw(gv(7), shoulder_ear_cm_right, "right")

        scr_cr = bucket(safe_average(cr_L, cr_R), *_B_CR)
        scr_imd = bucket(safe_average(imd_L, imd_R), *_B_IMD)
        scr_lsf = bucket(safe_average(lsf_L, lsf_R), *_B_LSF)
        scr_ttw = bucket(safe_average(ttw_L, ttw_R), *_B_TRAGUS)

        row = [pid, be.name] + [v if not np.isnan(v) else "NA" for v in [
            cr_L, cr_R, imd_L, imd_R, lsf_L, lsf_R, ttw_L, ttw_R,
            scr_cr, scr_imd, scr_lsf, scr_ttw,
            avg_runtime,
        ]]
        rows.append(row)
    return rows

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
        "avg_runtime_sec"
    ]

    with C.RESULTS_CSV.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for pid, rec in participants.items():
            for row in process_participant(pid, rec):
                wr.writerow(row)

if __name__ == "__main__":
    main()