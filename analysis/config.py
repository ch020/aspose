from pathlib import Path

# IO PATHS
ZIP_INPUT_DIR = Path(r"Zip")
EXTRACTED_OUTPUT_DIR = Path(r"Extracted")
RESULTS_CSV = Path(r"results.csv")

# RUNTIME OPTIONS
POSE_BACKENDS = {
    "cr": ["blazepose"],
    "lsf": ["blazepose", "hrnet", "openpose"],
    "imd": ["blazepose", "hrnet", "openpose"],
    "ttw": ["blazepose"],
}
SMOOTHING = True

SG_WINDOW_LENGTH = 9  # must be odd
SG_POLY_ORDER = 2
OUTLIER_ZSCORE = 3.0

assert SG_WINDOW_LENGTH % 2 == 1, "SG_WINDOW_LENGTH must be odd."

# WEIGHTS
WEIGHTS_DIR = Path(r"weights")