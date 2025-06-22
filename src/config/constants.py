from pathlib import Path

# ------------------------------------------------------------------ #
#  Project roots                                                     #
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/Cd_prediction/C_d
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
EXP_DIR = PROJECT_ROOT / "experiments"

# ------------------------------------------------------------------ #
#  Raw / processed paths                                             #
# ------------------------------------------------------------------ #
POINT_CLOUDS_DIR = RAW_DIR / "point_clouds"
SUBSET_DIR = RAW_DIR / "subset_dir"

SLICE_DIR = PROC_DIR / "slices"
PADDED_MASKED_SLICES_DIR = PROC_DIR / "padded_masked_slices"
SCALER_FILE = PROC_DIR / "scalars" / "scaler.pkl"  # <— single file

# ------------------------------------------------------------------ #
#  Pre-processing defaults                                           #
# ------------------------------------------------------------------ #
DEFAULT_NUM_SLICES = 80
DEFAULT_SLICE_AXIS = "x"
DEFAULT_TARGET_POINTS = 6_500
