"""
The main cli tool.
This is supposed to be run from the project's root.
All the paths are relative to the projects root.
"""

import argparse
import os

import numpy as np

from src.config.constants import (
    DEFAULT_NUM_SLICES,
    DEFAULT_SLICE_AXIS,
    DEFAULT_TARGET_POINTS,
    PADDED_MASKED_SLICES_DIR,
    POINT_CLOUDS_DIR,
    SLICE_DIR,
    SUBSET_DIR,
)
from src.data.slices import PointCloudSlicer, display_slices, process_all_slices
from src.evaluation.evaluate import run_evaluation
from src.inference.predict import run_inference
from src.training.ignite_loops import run_training


def main():
    parser = argparse.ArgumentParser(
        description="cd_prediction: slicing, training, evaluation, inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # slice subcommand
    slice_p = subparsers.add_parser("slice", help="Slice 3D point clouds")
    slice_p.add_argument("--input-dir", default=POINT_CLOUDS_DIR)
    slice_p.add_argument("--output-dir", default=SLICE_DIR)
    slice_p.add_argument("--num-slices", type=int, default=DEFAULT_NUM_SLICES)
    slice_p.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_SLICE_AXIS)
    slice_p.add_argument("--max-files", type=int)
    slice_p.add_argument("--split", choices=["train", "val", "test"], default="train")
    slice_p.add_argument("--subset-dir", default=SUBSET_DIR)

    # visualize subcommand
    viz_p = subparsers.add_parser(
        "visualize", help="Visualize saved 2D slices from a .npy file"
    )
    viz_p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to .npy file containing slices (object array)",
    )
    viz_p.add_argument(
        "--cols", type=int, default=5, help="Number of columns in grid display"
    )
    viz_p.add_argument(
        "--limit", type=int, default=None, help="Max number of slices to show"
    )
    viz_p.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default=DEFAULT_SLICE_AXIS,
        help="Axis the slices were taken along (for titles)",
    )
    viz_p.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="If provided, save figure here instead of showing",
    )

    # pad subcommand
    pad_p = subparsers.add_parser(
        "pad", help="Pad & mask 2D slices into fixed‐size arrays"
    )
    pad_p.add_argument(
        "--split",
        "-s",
        choices=["train", "val", "test"],
        required=True,
        help="Which split to process",
    )
    pad_p.add_argument(
        "--slice-dir", default=SLICE_DIR, help="Directory with .npy slice files"
    )
    pad_p.add_argument(
        "--output-dir",
        default=PADDED_MASKED_SLICES_DIR,
        help="Where to save .npz files",
    )
    pad_p.add_argument(
        "--target-slices",
        type=int,
        default=DEFAULT_NUM_SLICES,
        help="Number of slices per sample",
    )
    pad_p.add_argument(
        "--target-points",
        type=int,
        default=DEFAULT_TARGET_POINTS,
        help="Max points per slice",
    )

    # train subcommand - Not Implemented
    train_p = subparsers.add_parser("train", help="Train the model")
    # TODO: Add actual arguments instead of this placeholder.
    train_p.add_argument(
        "--config", type=str, required=True, help="Path to config yaml/json"
    )

    # eval subcommand - Not Implemented
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    # TODO: Add actual arguments instead of these placeholders.
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", choices=["val", "test"], default="test")

    # predict subcommand - Not Implemented
    pred_p = subparsers.add_parser("predict", help="Run inference")
    # TODO: Add actual arguments instead of these placeholders.
    pred_p.add_argument("--checkpoint", required=True)
    pred_p.add_argument("--input-data", required=True)
    pred_p.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "slice":
        slicer = PointCloudSlicer(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_slices=args.num_slices,
            axis=args.axis,
            max_files=args.max_files,
            split=args.split,
            subset_dir=args.subset_dir,
        )
        slicer.run()

    elif args.command == "visualize":
        path = args.input
        if not os.path.exists(path):
            parser.error(f"File not found: {path}")
        slices = np.load(path, allow_pickle=True)
        car_id = os.path.splitext(os.path.basename(path))[0]
        display_slices(
            slices,
            car_id=car_id,
            n_cols=args.cols,
            limit=args.limit,
            axis=args.axis,
            save_path=args.save_path,
        )

    elif args.command == "pad":
        process_all_slices(
            slice_dir=args.slice_dir,
            output_dir=args.output_dir,
            split=args.split,
            target_slices=args.target_slices,
            target_points=args.target_points,
        )

    elif args.command == "train":
        run_training()

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "predict":
        run_inference()


if __name__ == "__main__":
    main()
