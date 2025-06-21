"""
Utility method for input/output from the disk.
"""

from pathlib import Path

import joblib

from src.config.constants import SCALER_FILE, SUBSET_DIR
from src.utils.logger import logging as logger


def load_design_ids(split, subset_dir: Path = SUBSET_DIR):
    split_file = split_file = Path(subset_dir) / f"{split}_design_ids.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        ids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(ids)} design IDs for split: {split}")
    return ids


def save_scaler(scaler, path: Path = SCALER_FILE) -> Path:
    """
    Persist a fitted sklearn-style scaler (e.g. StandardScaler) to disk.

    Args:
        scaler: any object with sklearn's fit/transform attributes (has mean_, scale_, etc.).
        path: path to the scaler file

    Returns:
        The full path where the scaler was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler to {path}")
    return path


def load_scaler(path: Path = SCALER_FILE):
    """
    Load a previously saved scaler from disk.

    Args:
        path: path to the scaler file

    Returns:
        The deserialized scaler object (e.g. a StandardScaler with mean_/scale_ populated).

    Raises:
        FileNotFoundError if the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No scaler found at: {path}")
    scaler = joblib.load(path)
    logger.info(f"Loaded scaler from {path}")
    return scaler
