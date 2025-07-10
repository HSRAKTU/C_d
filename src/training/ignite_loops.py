"""
Training loop for Cd prediction.

Run from project root, e.g.:

    python -m src.main train \
        --config experiments/configuration.json \
        --resume experiments/checkpoints/best_model_val_loss=0.0123.pt

The file expects:
*   A CD dataset (`src.data.dataset.CdDataset`)
*   A model (`src.models.model.CdRegressor`)
*   Project-wide paths in `src.config.constants`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    ModelCheckpoint,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.metrics import MeanAbsoluteError, MeanSquaredError
from ignite.metrics.regression.r2_score import R2Score
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config.constants import (  # :contentReference[oaicite:2]{index=2}
    EXP_DIR,
    PREPARED_DATASET_DIR,
)
from src.data.dataset import CdDataset  # requires dataset.py implemented earlier

# from src.models.experiment_models.model_PTM import (
#     CdRegressor,
# )  # consolidated model module
from src.models.model_test import CdRegressor
from src.utils.io import load_config
from src.utils.logger import logger

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler


# TODO: Add support for early stopping
# TODO: Do training eval during the training, do not iterate the whole thing again.


def make_unscale(scaler: StandardScaler):
    def _unscale(x, y, y_pred):
        """
        This is used to unscale the output from the model before passing it on for
        metrics calculation. Ignite passes `(y_pred, y)` in the `output` argument.

        Args:
            output: A tuple of (y_pred, y)

        Returns:
            A tuple of unscaled prediction and real values. That is, (y_pred_u, y_u)
        """

        y_pred_u = (
            torch.from_numpy(
                scaler.inverse_transform(y_pred.detach().cpu().reshape(-1, 1))
            )
            .to(y_pred.device)
            .view_as(y_pred)
        )
        y_u = (
            torch.from_numpy(scaler.inverse_transform(y.detach().cpu().reshape(-1, 1)))
            .to(y.device)
            .view_as(y)
        )
        return y_pred_u, y_u

    return _unscale


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
    preapred_dataset_dir: Path = PREPARED_DATASET_DIR,
    fit_scaler: bool = False,
):
    """
    Main entry-point called from CLI to run training.
    The training config is loaded from `cfg_path`.
    The training is resumed if checkpoint path `resume` if provided.

    Args:
        exp_name: name of this experiment (used to create sub-directories for
        checkpoints and tb-logs)
        cfg_path: path to YAML / JSON describing the experiment.
        resume:   optional checkpoint to resume (full state: model, optimiser, trainer).
        preapred_dataset_dir: path to the directory with the prepared dataset.
    """
    cfg = load_config(cfg_path)

    # --------------------------------------------------------------------- #
    # Reproducibility & device                                              #
    # --------------------------------------------------------------------- #
    seed = cfg.get("seed", 42)
    debugging = cfg.get("debugging", False)
    torch.manual_seed(seed)
    device = _prepare_device(cfg.get("device"))

    # --------------------------------------------------------------------- #
    # Data                                                                  #
    # --------------------------------------------------------------------- #
    padded: bool = cfg["data"]["padded"]
    train_set = CdDataset(
        root_dir=preapred_dataset_dir,
        split="train",
        fit_scaler=fit_scaler,
        padded=padded,
        debugging=debugging,
    )
    val_set = CdDataset(
        root_dir=preapred_dataset_dir,
        split="val",
        fit_scaler=False,
        padded=padded,
        debugging=debugging,
    )
    scaler = train_set.scaler
    unscale_fn = make_unscale(scaler=scaler)

    if padded:
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
    else:
        pass
    # TODO: add PyG data loaders

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
    train_evaluator = create_supervised_evaluator(
        model,
        metrics={
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
        },
        output_transform=unscale_fn,
        device=device,
    )
    val_evaluator = create_supervised_evaluator(
        model,
        metrics={
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r2": R2Score(),
        },
        output_transform=unscale_fn,
        device=device,
    )
    # Loss calculated by trainer is scaled. Only for training and monitoring purpose.
    # Metrics calculated by evaluators are unscaled.

    # --------------------------------------------------------------------- #
    # Attach Progress Bar                                                   #
    # --------------------------------------------------------------------- #
    train_pbar = ProgressBar(desc="Training", persist=True)
    train_pbar.attach(trainer, output_transform=lambda loss: {"batch_loss": loss})

    train_eval_pbar = ProgressBar(desc="Training Set Evaluation", persist=True)
    train_eval_pbar.attach(train_evaluator)
    val_eval_pbar = ProgressBar(desc="Validation Set Evaluation", persist=True)
    val_eval_pbar.attach(val_evaluator)

    # Log running loss every N iterations
    log_interval = cfg["training"].get("log_interval", 100)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def _log_iter(engine):
        train_pbar.log_message(
            f"\n\nEpoch[{engine.state.epoch}] "
            f"Iter[{engine.state.iteration}] "
            f"bach_loss={engine.state.output:.4f}\n\n"  # scaled batch loss for this iteration
        )

    # Evaluate & checkpoint every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval_and_log(engine):
        train_evaluator.run(train_loader)
        train_metrics = train_evaluator.state.metrics
        val_evaluator.run(val_loader)
        val_metrics = val_evaluator.state.metrics
        train_pbar.log_message(
            f"\n\nEpoch {engine.state.epoch}: "
            f"train MAE={train_metrics['mae']:.4f} | "
            f"train R2={train_metrics['r2']:.4f} | "
            f"val MAE={val_metrics['mae']:.4f} | "
            f"val R2={val_metrics['r2']:.4f}\n\n"
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
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["mae", "mse", "r2"],
        global_step_transform=global_step_from_engine(trainer),
    )

    # --------------------------------------------------------------------- #
    # Checkpointing                                                         #
    # --------------------------------------------------------------------- #
    def score_fn(eng):
        return -eng.state.metrics["mae"]  # minimise MAE

    # save the best models based on val_evaluator's mae
    # (meant for inference, not resuming training)
    best_model_saver = ModelCheckpoint(
        dirname=EXP_DIR / exp_name / "model_checkpoints",
        filename_prefix="best_model_",
        n_saved=3,
        global_step_transform=global_step_from_engine(trainer),
        score_function=score_fn,
        score_name="val_mae",
        require_empty=False,
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED, best_model_saver, {"model": model}
    )

    # save the entire training state after every epoch of trainer
    # (meant for resuming)
    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer}
    training_state_saver = Checkpoint(
        to_save=to_save,
        save_handler=DiskSaver(
            EXP_DIR / exp_name / "training_state_checkpoints",
            create_dir=True,
            atomic=True,
        ),
        filename_prefix="training_state_",
        global_step_transform=global_step_from_engine(trainer),
        n_saved=5,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, training_state_saver)

    # Resume full state if requested
    if resume:
        ckpt_fp = Path(resume)
        if not ckpt_fp.exists():
            logger.error(f"Resume checkpoint not found: {ckpt_fp}")
            sys.exit(1)
        logger.info(f"Resuming from checkpoint: {ckpt_fp}")
        to_load = {"model": model, "optimizer": optimizer, "trainer": trainer}
        Checkpoint.load_objects(
            to_load=to_load, checkpoint=torch.load(ckpt_fp, map_location=device)
        )
        # TODO: add the ability to overwrite learning rate of the optimizer

    # --------------------------------------------------------------------- #
    # Start training                                                        #
    # --------------------------------------------------------------------- #
    with logging_redirect_tqdm():
        trainer.run(train_loader, max_epochs=cfg["training"]["epochs"])
    tb_logger.close()
