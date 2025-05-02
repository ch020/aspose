from pathlib import Path

# IO PATHS
ZIP_INPUT_DIR = Path(r"Zip")  # Path containing zipped data
EXTRACTED_OUTPUT_DIR = Path(r"Extracted")  # Path to extract zipped data to
# RESULTS_CSV = Path(r"results.csv") 

# RUNTIME OPTIONS
POSE_BACKENDS_TO_RUN = ["blazepose", "openpose", "hrnet"]  # Pose Backends to Run
POSE_BACKENDS = {  # Dictionary containing each BASMI metric and the models that can calculate it
    "cr": ["blazepose"],
    "lsf": ["blazepose", "hrnet", "openpose"],
    "imd": ["blazepose", "hrnet", "openpose"],
    "ttw": ["blazepose"],
}
OUTPUT_CSV = Path("output.csv")  # The output CSV path
SMOOTHING = True  # Whether to apply a Savitzky–Golay filter to the data

SG_WINDOW_LENGTH = 9  # must be odd, used for smoothing
SG_POLY_ORDER = 2  # used for smoothing
OUTLIER_ZSCORE = 3.0  # zscore for outlier removal

assert SG_WINDOW_LENGTH % 2 == 1, "SG_WINDOW_LENGTH must be odd."

# HRNET
HRNET_ROOT = Path(r"hrnet")  # Root HRNET directory

# OPENPOSE
OPENPOSE_ROOT = Path("openpose")  # Root OpenPose Directory
OPENPOSE_BIN = Path(OPENPOSE_ROOT / "bin")  # Path to OpenPose binaries folder (containing OpenPoseDemo.exe)
