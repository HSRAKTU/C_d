"""
Utility method for input/output from the disk.
"""

import os

import joblib

from src.config.constants import SCALER_DIR, SUBSET_DIR
from src.utils.logger import logging as logger


def load_design_ids(split, subset_dir=SUBSET_DIR):
    split_file = os.path.join(subset_dir, f"{split}_design_ids.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        ids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(ids)} design IDs for split: {split}")
    return ids


def save_scaler(
    scaler, save_dir: str = SCALER_DIR, filename: str = "scaler.pkl"
) -> str:
    """
    Persist a fitted sklearn-style scaler (e.g. StandardScaler) to disk.

    Args:
        scaler: any object with sklearn's fit/transform attributes (has mean_, scale_, etc.).
        save_dir: directory under which to save the scaler file.
        filename: name of the file (should end in .pkl).

    Returns:
        The full path where the scaler was written.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler to {path}")
    return path


def load_scaler(load_dir: str = SCALER_DIR, filename: str = "scaler.pkl"):
    """
    Load a previously saved scaler from disk.

    Args:
        load_dir: directory where the scaler file lives.
        filename: name of the file to load.

    Returns:
        The deserialized scaler object (e.g. a StandardScaler with mean_/scale_ populated).

    Raises:
        FileNotFoundError if the file does not exist.
    """
    path = os.path.join(load_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No scaler found at: {path}")
    scaler = joblib.load(path)
    logger.info(f"Loaded scaler from {path}")
    return scaler
