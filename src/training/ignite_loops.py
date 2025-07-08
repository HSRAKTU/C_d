"""
Ignite-powered training loop for Cd prediction.

Run from project root, e.g.:

    python -m src.main train \
        --config experiments/baseline.yaml \
        --resume experiments/checkpoints/best_model_val_loss=0.0123.pt

The file expects:
*   A CD dataset (`src.data.dataset.CdDataset`)
*   A model (`src.models.model.CdRegressor`)
*   Project-wide paths in `src.config.constants`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
import yaml
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import (
    Checkpoint,
    ModelCheckpoint,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
from ignite.metrics.regression.r2_score import R2Score
from torch.utils.data import DataLoader

from src.config.constants import (  # :contentReference[oaicite:2]{index=2}
    EXP_DIR,
    PREPARED_DATASET_DIR,
)
from src.data.dataset import CdDataset  # requires dataset.py implemented earlier
from src.models.experiment_models.model_PTM import (
    CdRegressor,
)  # consolidated model module
from src.utils.logger import logger


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_config(cfg_path: Union[str, Path]) -> Dict:
    """Load a YAML or JSON experiment-config file."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if cfg_path.suffix in {".yml", ".yaml"}:
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)
    elif cfg_path.suffix == ".json":
        with cfg_path.open() as f:
            cfg = json.load(f)
    else:
        raise ValueError(f"Unsupported config type: {cfg_path.suffix}")
    logger.info(f"Loaded config from {cfg_path}")
    return cfg


def _prepare_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_training(
    exp_name: str,
    cfg_path: str | Path,
    resume: Path | None = None,
    batch_size: int | None = None,
):
    """
    Main entry-point called from CLI.

    Args:
        exp_name: name of this experiment (used to create sub-directories for checkpoints and tb-logs)
        cfg_path: path to YAML / JSON describing the experiment.
        resume:   optional checkpoint to resume (full state: model, optimiser, trainer).
    """
    cfg = _load_config(cfg_path)

    # --------------------------------------------------------------------- #
    # Overrides from CLI                                                    #
    # --------------------------------------------------------------------- #
    if batch_size:
        cfg["data"]["batch_size"] = batch_size
    if resume:
        cfg["training"]["resume"] = resume

    # --------------------------------------------------------------------- #
    # Reproducibility & device                                              #
    # --------------------------------------------------------------------- #
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    device = _prepare_device(cfg.get("device"))

    # --------------------------------------------------------------------- #
    # Data                                                                  #
    # --------------------------------------------------------------------- #
    first_run = cfg["training"].get("resume") is None
    train_set = CdDataset(
        root_dir=PREPARED_DATASET_DIR,
        split="train",
        fit_scaler=first_run,  # fit & save only on first run
    )
    val_set = CdDataset(
        root_dir=PREPARED_DATASET_DIR,
        split="val",
        fit_scaler=False,  # always load
    )
    # share the very same scaler object to avoid second disk read
    val_set.scaler = train_set.scaler

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["data"].get("batch_size", 4),
        pin_memory=(device.type == "cuda"),
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["data"].get("batch_size", 4),
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
    )

    # --------------------------------------------------------------------- #
    # Model, optimiser, criterion                                           #
    # --------------------------------------------------------------------- #
    model = CdRegressor(**cfg["model"]).to(device)
    optim_params = cfg.get("optim", {}).get("params", {"lr": 1e-3})
    optimizer = torch.optim.Adam(model.parameters(), **optim_params)
    criterion = nn.MSELoss()

    # --------------------------------------------------------------------- #
    # Ignite engines                                                        #
    # --------------------------------------------------------------------- #
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
        },
        device=device,
    )

    # Log running loss every N iterations
    log_interval = cfg["training"].get("log_interval", 100)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def _log_iter(engine):
        logger.info(
            f"Epoch[{engine.state.epoch}] "
            f"Iter[{engine.state.iteration}] "
            f"loss={engine.state.output:.4f}"
        )

    # Evaluate & checkpoint every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval_and_log(engine):
        evaluator.run(train_loader)
        train_metrics = evaluator.state.metrics
        evaluator.run(val_loader)
        val_metrics = evaluator.state.metrics
        logger.info(
            f"Epoch {engine.state.epoch}: "
            f"train MAE={train_metrics['mae']:.4f} "
            f"train R2={train_metrics['r2']:.4f} "
            f"val MAE={val_metrics['mae']:.4f}"
            f"val R2={val_metrics['r2']:.4f}"
        )

    # --------------------------------------------------------------------- #
    # TensorBoard                                                           #
    # --------------------------------------------------------------------- #
    tb_logger = TensorboardLogger(log_dir=EXP_DIR / exp_name / "tb-logs")
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["mae", "mse", "r2"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # --------------------------------------------------------------------- #
    # Checkpointing & (optional) resume                                     #
    # --------------------------------------------------------------------- #
    def score_fn(eng):
        return -eng.state.metrics["mae"]  # minimise MAE

    saver = ModelCheckpoint(
        dirname=EXP_DIR / exp_name / "checkpoints",
        filename_prefix="best",
        n_saved=3,
        global_step_transform=global_step_from_engine(trainer),
        score_function=score_fn,
        score_name="val_mae",
        require_empty=False,
    )
    evaluator.add_event_handler(Events.COMPLETED, saver, {"model": model})

    # Resume full state if requested
    if cfg["training"].get("resume"):
        ckpt_fp = Path(cfg["training"]["resume"])
        if not ckpt_fp.exists():
            logger.error(f"Resume checkpoint not found: {ckpt_fp}")
            sys.exit(1)
        logger.info(f"Resuming from checkpoint: {ckpt_fp}")
        to_load = {"model": model, "optimizer": optimizer, "trainer": trainer}
        Checkpoint.load_objects(
            to_load=to_load, checkpoint=torch.load(ckpt_fp, map_location=device)
        )

    # --------------------------------------------------------------------- #
    # Attach Progress Bar                                                   #
    # --------------------------------------------------------------------- #
    train_pbar = ProgressBar(desc="Training", persist=True)
    train_pbar.attach(trainer, output_transform=lambda loss: {"batch_loss": loss})

    val_pbar = ProgressBar(desc="Validation", persist=True)
    val_pbar.attach(evaluator)

    # --------------------------------------------------------------------- #
    # Start training                                                        #
    # --------------------------------------------------------------------- #
    trainer.run(train_loader, max_epochs=cfg["training"]["epochs"])
    tb_logger.close()
