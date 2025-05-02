from pathlib import Path

# IO PATHS
ZIP_INPUT_DIR = Path(r"Zip")
EXTRACTED_OUTPUT_DIR = Path(r"Extracted")
RESULTS_CSV = Path(r"results.csv")

# RUNTIME OPTIONS
POSE_BACKENDS_TO_RUN = ["blazepose", "openpose", "hrnet"]
POSE_BACKENDS = {
    "cr": ["blazepose"],
    "lsf": ["blazepose", "hrnet", "openpose"],
    "imd": ["blazepose", "hrnet", "openpose"],
    "ttw": ["blazepose"],
}
OUTPUT_CSV = Path("output.csv")
SMOOTHING = True

SG_WINDOW_LENGTH = 9  # must be odd
SG_POLY_ORDER = 2
OUTLIER_ZSCORE = 3.0

assert SG_WINDOW_LENGTH % 2 == 1, "SG_WINDOW_LENGTH must be odd."

# HRNET
HRNET_ROOT = Path(r"hrnet")

# OPENPOSE
OPENPOSE_ROOT = Path("openpose")
OPENPOSE_BIN = Path(OPENPOSE_ROOT / "bin")
