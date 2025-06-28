"""
Loads .paddle_tensor point clouds, bins them into num_slices along axis.
Pads the slices to target number of points per slice.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import paddle
from tqdm import tqdm

from src.config.constants import (
    DEFAULT_NUM_SLICES,
    DEFAULT_SLICE_AXIS,
    DEFAULT_TARGET_POINTS,
    SUBSET_DIR
)
from src.utils.io import load_cd_map, load_design_ids
from src.utils.logger import logger


class PointCloudSlicer:
    def __init__(
        self,
        input_dir,
        output_dir,
        num_slices,
        axis,
        max_files,
        split,
        subset_dir,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_slices = num_slices
        self.axis = axis
        self.max_files = max_files
        self.split = split
        self.subset_dir = Path(subset_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.axis_map = {"x": 0, "y": 1, "z": 2}
        if self.axis not in self.axis_map:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

        if self.split == "all":
            self.valid_ids = load_design_ids("train", self.subset_dir) | load_design_ids(
                "val", self.subset_dir
            ) | load_design_ids("test", self.subset_dir)
        else:
            self.valid_ids = load_design_ids(self.split, self.subset_dir)

    def load_point_cloud(self, file_path: Path):
        tensor = paddle.load(str(file_path))
        return tensor.numpy()

    def generate_slices(self, points):
        """
        Evenly divides the chosen axis range into num_slices bins and returns a
        list of (Nᵢ, 2) arrays where axis dimension is dropped.
        """
        ax = self.axis_map[self.axis]
        coords = points[:, ax]
        min_val, max_val = coords.min(), coords.max()
        edges = np.linspace(min_val, max_val, self.num_slices + 1)

        slices = []
        for i in range(self.num_slices):
            low, high = edges[i], edges[i + 1]
            mask = (coords >= low) & (coords < high)
            sl = points[mask]
            sl = np.delete(sl, ax, axis=1)
            slices.append(sl)
        return slices

    def process_file(self, file_path: Path):
        points = self.load_point_cloud(file_path)
        slices = self.generate_slices(points)
        total_points = sum(len(sl) for sl in slices)
        logger.info(
            f"{file_path.name} → {total_points} total points across {len(slices)} slices."
        )
        return slices

    def run(self):
        all_files = sorted(self.input_dir.glob("*.paddle_tensor"))
        filtered_files = [f for f in all_files if f.stem in self.valid_ids]

        if self.max_files is not None:
            filtered_files = filtered_files[: self.max_files]

        logger.info(
            f"Processing {len(filtered_files)} point clouds from split: {self.split}"
        )

        count_success, count_error = 0, 0

        for file_path in tqdm(filtered_files, desc=f"Slicing {self.split} set"):
            try:
                car_id = file_path.stem
                output_path = self.output_dir / f"{car_id}_axis-{self.axis}.npy"
                slices = self.process_file(file_path)
                np.save(output_path, np.array(slices, dtype=object), allow_pickle=True)
                count_success += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                count_error += 1

        logger.info(
            f"Finished slicing split '{self.split}': {count_success} succeeded, {count_error} failed."
        )


def pad_and_mask_slices(
    slice_list, target_slices=DEFAULT_NUM_SLICES, target_points=DEFAULT_TARGET_POINTS
):
    padded = np.zeros((target_slices, target_points, 2), dtype=np.float32)
    point_mask = np.zeros((target_slices, target_points), dtype=np.float32)
    slice_mask = np.zeros((target_slices,), dtype=np.float32)

    for i, sl in enumerate(slice_list):
        if sl.shape[0] == 0:
            continue
        n_pts = min(sl.shape[0], target_points)
        padded[i, :n_pts] = sl[:n_pts]
        point_mask[i, :n_pts] = 1
        slice_mask[i] = 1

    return padded, point_mask, slice_mask


def process_all_slices(
    slice_dir,
    output_dir,
    split,
    target_slices=DEFAULT_NUM_SLICES,
    target_points=DEFAULT_TARGET_POINTS,
    subset_dir=SUBSET_DIR,
):
    slice_dir = Path(slice_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if split == "all":
        design_ids = load_design_ids("train", subset_dir) | load_design_ids("val", subset_dir) | load_design_ids("test", subset_dir)
    else:
        design_ids = load_design_ids(split, subset_dir)
    cd_table = load_cd_map()
    logger.info(f"Preparing split: {split} → {len(design_ids)} design IDs")

    all_files = [f for f in slice_dir.glob("*.npy")]
    matched = [f for f in all_files if f.stem.split("_axis")[0] in design_ids]
    logger.info(f"Matched {len(matched)} files in {slice_dir} for split '{split}'")

    for fname in tqdm(matched, desc=f"Pad/mask {split}", ncols=80):
        in_path = fname
        try:
            slices = np.load(in_path, allow_pickle=True)
            padded, pmask, smask = pad_and_mask_slices(
                slices, target_slices, target_points
            )
            car_id = fname.stem
            design_id = car_id.split("_axis")[0]
            cd_val = cd_table.get(design_id)
            if cd_val is None:
                logger.warning(f"Cd not found for {design_id} – file skipped")
                continue

            out_path = output_dir / f"{car_id}.npz"
            np.savez_compressed(
                out_path,
                slices=padded,
                point_mask=pmask,
                slice_mask=smask,
                Cd=cd_val,
            )
        except Exception as e:
            logger.warning(f"{fname.name} failed: {e}")

    logger.info(f"Done: {len(matched)} → {output_dir}")


def display_slices(
    slices,
    car_id=None,
    n_cols=5,
    limit=None,
    figsize=(15, 3),
    axis=DEFAULT_SLICE_AXIS,
    save_path=None,
):
    """
    Display or save 2D slices from a point cloud.

    slices: list of (num_of_points, 2) np arrays
    """
    if limit:
        slices = slices[:limit]

    all_points = np.vstack([sl for sl in slices if sl.size])
    xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
    ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()

    pad_x = 0.02 * (xmax - xmin)
    pad_y = 0.02 * (ymax - ymin)
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y

    n = len(slices)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    for idx, sl in enumerate(slices):
        ax = axes.flat[idx]
        if sl.size:
            ax.scatter(sl[:, 0], sl[:, 1], s=2, c="k")
        else:
            ax.text(0.5, 0.5, "Empty", ha="center", va="center")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(f"{axis} ∈ slice {idx}", fontsize=8)

    for j in range(n, n_rows * n_cols):
        axes.flat[j].axis("off")

    fig.suptitle(
        f"Slices for {car_id}" if car_id else "Point Cloud Slices", fontsize=14
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[✓] Saved to {save_path}")
        plt.close()
    else:
        plt.show()
