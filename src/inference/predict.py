"""
Batch inference for Cd prediction.

Usage
-----
python -m src.main predict \
    --config experiments/baseline.yaml \
    --checkpoint experiments/checkpoints/best_model_val_mae=0.0123.pt \
    --input-data data/samples_to_score              # file OR directory of *.npz
    --output predictions.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader, Dataset

from src.config.constants import SCALER_FILE
from src.models.model import CdRegressor
from src.utils.io import load_scaler
from src.utils.logger import logging as logger


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_config(cfg_path: Union[str, Path]) -> Dict:
    """Load YAML / JSON experiment config."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if cfg_path.suffix in {".yml", ".yaml"}:
        return yaml.safe_load(cfg_path.read_text())
    if cfg_path.suffix == ".json":
        return json.loads(cfg_path.read_text())
    raise ValueError(f"Unsupported config: {cfg_path}")


def _prepare_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class InferenceDataset(Dataset):
    """
    Loads *.npz files containing:
        slices     -> (S, P, 2)
        point_mask -> (S, P)
        slice_mask -> (S,)
    """

    def __init__(self, files: Sequence[Path]):
        self.files = list(files)
        self.scaler = load_scaler(SCALER_FILE)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        fp = self.files[idx]
        arr = dict(
            **torch.load(fp)
            if fp.suffix == ".pt"
            else dict(np.load(fp, allow_pickle=True))
        )
        slices = torch.as_tensor(arr["slices"]).float()
        slices = torch.as_tensor(
            self.scaler.transform(slices.reshape(-1, 2)).reshape(slices.shape)
        ).float()
        p_mask = torch.as_tensor(arr["point_mask"]).float()
        s_mask = torch.as_tensor(arr["slice_mask"]).float()
        design_id = fp.stem
        return slices, p_mask, s_mask, design_id


def _collate(batch):
    """Keeps design_ids in a list to avoid default tensor stacking."""
    slices, p_masks, s_masks, ids = zip(*batch)
    return (
        torch.stack(slices, dim=0),
        torch.stack(p_masks, dim=0),
        torch.stack(s_masks, dim=0),
        list(ids),
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_inference(
    cfg_path: str | Path,
    checkpoint_path: str | Path,
    input_data: str | Path,
    output_path: str | Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> List[Tuple[str, float]]:
    """
    Generate Cd predictions for a set of *.npz slice files.

    Returns list of (design_id, Cd) tuples and writes a CSV.
    """
    cfg = _load_config(cfg_path)
    device = _prepare_device(cfg.get("device"))

    # --------------------------- files to score -------------------------- #
    input_path = Path(input_data)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npz"))
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(input_path)
    if not files:
        raise RuntimeError(f"No .npz files found in {input_path}")

    ds = InferenceDataset(files)
    dl = DataLoader(
        ds,
        batch_size=batch_size or cfg["data"].get("batch_size", 8),
        num_workers=num_workers
        if num_workers is not None
        else cfg["data"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        collate_fn=_collate,
    )

    # ----------------------------- model ---------------------------------- #
    model = CdRegressor(**cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint {checkpoint_path}")

    # --------------------- Ignite inference engine ------------------------ #
    predictions: List[Tuple[str, float]] = []

    def _step(engine, batch):
        slices, p_mask, s_mask, ids = batch
        slices, p_mask, s_mask = slices.to(device), p_mask.to(device), s_mask.to(device)
        with torch.no_grad():
            preds = model((slices, p_mask, s_mask)).squeeze()
        preds = preds.cpu().tolist()
        if isinstance(preds, float):  # batch_size == 1
            preds = [preds]
        return list(zip(ids, preds))

    infer_engine = Engine(_step)

    @infer_engine.on(Events.ITERATION_COMPLETED)
    def _gather(engine):
        predictions.extend(engine.state.output)

    infer_engine.run(dl)

    # ---------------------------- save CSV -------------------------------- #
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["design_id", "Cd"])
        writer.writerows(predictions)
    logger.info(f"Saved {len(predictions)} predictions → {output_path}")

    return predictions
