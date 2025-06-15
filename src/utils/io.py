"""
Utility method for input/output from the disk.
"""

import os

from src.utils.logger import logger


def load_design_ids(split, subset_dir="data/raw/subset_dir"):
    split_file = os.path.join(subset_dir, f"{split}_design_ids.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        ids = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(ids)} design IDs for split: {split}")
    return ids
