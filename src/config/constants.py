import os

# PATHS
BASE_DATA_DIR = "data"

RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw")
POINT_CLOUDS_DIR = os.path.join(RAW_DATA_DIR, "point_clouds")
SUBSET_DIR = os.path.join(RAW_DATA_DIR, "subset_dir")

PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, "processed")
SCALER_DIR = os.path.join(PROCESSED_DATA_DIR, "scalars")
SLICE_DIR = os.path.join(PROCESSED_DATA_DIR, "slices")
PADDED_MASKED_SLICES_DIR = os.path.join(PROCESSED_DATA_DIR, "padded_masked_slices")


CHECKPOINT_DIR = "experiments/checkpoints"
TB_LOG_DIR = "experiments/tb-logs"

# PREPROCESSING
DEFAULT_NUM_SLICES = 80
DEFAULT_SLICE_AXIS = "x"
DEFAULT_TARGET_POINTS = 6500
