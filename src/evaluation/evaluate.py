"""
Run offline evaluation of a saved Cd-Regressor checkpoint.

Example
-------
python -m src.main evaluate \
    --config experiments/baseline.yaml \
    --checkpoint experiments/exp_name/checkpoints/best_model_val_mae=0.0123.pt \
    --split test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import torch
import yaml
from ignite.engine import create_supervised_evaluator
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
from ignite.metrics.regression.r2_score import R2Score
from torch.utils.data import DataLoader

from src.config.constants import PADDED_MASKED_SLICES_DIR
from src.data.dataset import CdDataset
from src.models.experiment_models.model_PTM import CdRegressor
from src.utils.logger import logger


# --------------------------------------------------------------------------- #
# Helpers (duplicated in ignite_loops.py – kept here for standalone use)      #
# --------------------------------------------------------------------------- #
def _load_config(cfg_path: Union[str, Path]) -> Dict:
    """Load YAML or JSON experiment description."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if cfg_path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(cfg_path.read_text())
    if cfg_path.suffix == ".json":
        return json.loads(cfg_path.read_text())
    raise ValueError(f"Unsupported config extension: {cfg_path.suffix}")


def _prepare_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_evaluation(
    cfg_path: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
    batch_size: int | None = None,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint on a dataset split.

    Args:
        cfg_path:        YAML / JSON used during training (to recreate model).
        checkpoint_path: Path to *.pt checkpoint produced by Ignite.
        split:           Dataset split – "val" or "test".
        batch_size:      Optional override.

    Returns:
        Dict with metric names → values.
    """
    cfg = _load_config(cfg_path)
    device = _prepare_device(cfg.get("device"))

    # ------------------------------- data -------------------------------- #
    ds = CdDataset(
        root_dir=PADDED_MASKED_SLICES_DIR,
        split=split,
        fit_scaler=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size or cfg["data"].get("batch_size", 8),
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------ model -------------------------------- #
    model = CdRegressor(**cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # ModelCheckpoint saved {"model": model} so look for that key first
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # ---------------------------- evaluator ------------------------------ #
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
        },
        device=device,
    )

    # ── Attach Evaluation Progress Bar ─────────────────────────────
    eval_pbar = ProgressBar(desc=f"Evaluating ({split})", persist=True)
    eval_pbar.attach(evaluator)

    # run once
    evaluator.run(dl)
    metrics = evaluator.state.metrics
    metrics["rmse"] = metrics["mse"] ** 0.5
    logger.info(
        f"Split={split} | "
        f"MAE={metrics['mae']:.4f} MSE={metrics['mse']:.4f} "
        f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}"
    )

    return metrics
