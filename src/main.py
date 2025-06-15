"""
Main command-line entry-point for the Cd-prediction project.

Run from the project root, e.g.

    python -m src.main train \
        --config experiments/baseline.yaml \
        --resume experiments/checkpoints/best_model_val_mae=0.012.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from src.utils.logger import logging as logger


# --------------------------------------------------------------------------- #
# CLI factory                                                                 #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Configure argparse sub-commands."""
    parser = argparse.ArgumentParser(
        description="Cd-prediction: slicing · training · evaluation · inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --------------------------------------------------------------------- #
    # slice                                                                 #
    # --------------------------------------------------------------------- #
    slice_p = subparsers.add_parser("slice", help="Slice 3-D point clouds")
    slice_p.add_argument("--input-dir", default=POINT_CLOUDS_DIR)
    slice_p.add_argument("--output-dir", default=SLICE_DIR)
    slice_p.add_argument("--num-slices", type=int, default=DEFAULT_NUM_SLICES)
    slice_p.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_SLICE_AXIS)
    slice_p.add_argument("--max-files", type=int)
    slice_p.add_argument("--split", choices=["train", "val", "test"], default="train")
    slice_p.add_argument("--subset-dir", default=SUBSET_DIR)

    # --------------------------------------------------------------------- #
    # visualize                                                             #
    # --------------------------------------------------------------------- #
    viz_p = subparsers.add_parser("visualize", help="Show saved 2-D slices")
    viz_p.add_argument("-i", "--input", required=True, help="Path to .npy slice file")
    viz_p.add_argument("--cols", type=int, default=5)
    viz_p.add_argument("--limit", type=int)
    viz_p.add_argument("--axis", choices=["x", "y", "z"], default=DEFAULT_SLICE_AXIS)
    viz_p.add_argument("--save-path")

    # --------------------------------------------------------------------- #
    # pad                                                                   #
    # --------------------------------------------------------------------- #
    pad_p = subparsers.add_parser("pad", help="Pad & mask slice arrays")
    pad_p.add_argument("-s", "--split", choices=["train", "val", "test"], required=True)
    pad_p.add_argument("--slice-dir", default=SLICE_DIR)
    pad_p.add_argument("--output-dir", default=PADDED_MASKED_SLICES_DIR)
    pad_p.add_argument("--target-slices", type=int, default=DEFAULT_NUM_SLICES)
    pad_p.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS)

    # --------------------------------------------------------------------- #
    # train                                                                 #
    # --------------------------------------------------------------------- #
    train_p = subparsers.add_parser("train", help="Train a model")
    train_p.add_argument(
        "--config", required=True, help="Path to YAML / JSON experiment config"
    )
    train_p.add_argument(
        "--resume",
        help="Full path to checkpoint to resume from.  "
        "If omitted, you will be prompted interactively.",
    )
    train_p.add_argument("--batch-size", type=int, help="Override batch size")
    train_p.add_argument("--num-workers", type=int, help="Override DataLoader workers")

    # --------------------------------------------------------------------- #
    # evaluate                                                              #
    # --------------------------------------------------------------------- #
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_p.add_argument("--config", required=True, help="Experiment config YAML / JSON")
    eval_p.add_argument("--checkpoint", required=True, help="Path to model *.pt file")
    eval_p.add_argument("--split", choices=["val", "test"], default="test")
    eval_p.add_argument("--batch-size", type=int)
    eval_p.add_argument("--num-workers", type=int)

    # --------------------------------------------------------------------- #
    # predict                                                               #
    # --------------------------------------------------------------------- #
    pred_p = subparsers.add_parser("predict", help="Run inference")
    pred_p.add_argument("--config", required=True, help="Experiment config YAML / JSON")
    pred_p.add_argument("--checkpoint", required=True, help="Model *.pt checkpoint")
    pred_p.add_argument(
        "--input-data", required=True, help="File or directory of *.npz"
    )
    pred_p.add_argument("--output", required=True, help="CSV path for predictions")
    pred_p.add_argument("--batch-size", type=int)
    pred_p.add_argument("--num-workers", type=int)

    return parser


# --------------------------------------------------------------------------- #
# Entrypoint dispatcher                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------ slice --------------------------------- #
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

    # ---------------------------- visualize ------------------------------- #
    elif args.command == "visualize":
        if not os.path.exists(args.input):
            parser.error(f"File not found: {args.input}")
        slices = np.load(args.input, allow_pickle=True)
        car_id = Path(args.input).stem
        display_slices(
            slices,
            car_id=car_id,
            n_cols=args.cols,
            limit=args.limit,
            axis=args.axis,
            save_path=args.save_path,
        )

    # ------------------------------- pad ---------------------------------- #
    elif args.command == "pad":
        process_all_slices(
            slice_dir=args.slice_dir,
            output_dir=args.output_dir,
            split=args.split,
            target_slices=args.target_slices,
            target_points=args.target_points,
        )

    # ------------------------------ train --------------------------------- #
    elif args.command == "train":
        resume = args.resume
        if resume is None:
            # Interactive prompt (require checkpoint) – complies with the user's spec
            resume = input(
                "🔄  Please provide a checkpoint path to resume from "
                "(or press Enter to cancel): "
            ).strip()
            if not resume:
                logger.error("No checkpoint supplied – training aborted.")
                sys.exit(1)

        if not Path(resume).is_file():
            logger.error(f"Checkpoint not found: {resume}")
            sys.exit(1)

        run_training(
            cfg_path=args.config,
            resume=resume,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ---------------------------- evaluate -------------------------------- #
    elif args.command == "evaluate":
        run_evaluation(
            cfg_path=args.config,
            checkpoint_path=args.checkpoint,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ----------------------------- predict -------------------------------- #
    elif args.command == "predict":
        run_inference(
            cfg_path=args.config,
            checkpoint_path=args.checkpoint,
            input_data=args.input_data,
            output_path=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
