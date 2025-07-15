"""
Print a detailed summary and size for the Cd-DLM model
------------------------------------------------------
$ python print_model_stats.py --config default_config.json --device cuda:0
$ python print_model_stats.py --device cpu                           # uses default config
"""

from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

# — project imports —
from src.models.model import get_model
from src.utils.io import load_config


# ------------------------------------------------------------------- #
# helpers                                                             #
# ------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module) -> SimpleNamespace:
    """
    Returns:
        total         : int
        trainable     : int
        per_component : dict[str, int]
    """
    total = trainable = 0
    per_component: dict[str, int] = defaultdict(int)

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        # e.g. "slice_encoder.conv1.weight"  ->  "slice_encoder"
        top = name.split(".")[0]
        per_component[top] += n
    return SimpleNamespace(
        total=total, trainable=trainable, per_component=dict(per_component)
    )


def estimate_state_dict_size_mb(model: torch.nn.Module) -> float:
    """
    Serialises state_dict to an in-memory buffer and returns its size (MiB).
    """
    buffer = io.BytesIO()
    torch.save(
        model.state_dict(), buffer
    )  # ← identical to saving to disk :contentReference[oaicite:0]{index=0}
    return buffer.getbuffer().nbytes / 1_048_576


# ------------------------------------------------------------------- #
# main                                                                #
# ------------------------------------------------------------------- #
def main(config: Path | str, device: str):
    # -------- config + model -------- #
    cfg = load_config(config)
    model_type = cfg["model"]["model_type"]
    model_params = cfg["model"][model_type]
    model = get_model(model_type="dlm", **model_params).to(device)

    # -------- parameter stats -------- #
    stats = count_parameters(
        model
    )  # Keras-style breakdown :contentReference[oaicite:1]{index=1}

    print("=" * 60)
    print(model)  # native PyTorch repr
    print("-" * 60)

    print(f"Total parameters     : {stats.total:,}")
    print(f"Trainable parameters : {stats.trainable:,}\n")

    print("Parameters by component")
    for comp, n in stats.per_component.items():
        print(f"  • {comp:<18}: {n:,}")

    # -------- size on disk -------- #
    size_mb = estimate_state_dict_size_mb(model)  # uses torch.save() buffer
    print(f"\nApprox. state_dict size: {size_mb:.2f} MiB")

    # -------- optional rich summary -------- #
    try:
        from torchinfo import (
            summary,
        )  # GitHub: TylerYep/torchinfo :contentReference[oaicite:2]{index=2}

        dummy_input = model.example_input(  # synthetic PyG batches accepted by summary
            batch_size=2, S=model.num_slices
        )  # torch_geometric Batch API :contentReference[oaicite:3]{index=3}

        summary(
            model,
            input_data=dummy_input,  # torchinfo handles list / tuple inputs :contentReference[oaicite:4]{index=4}
            depth=3,
            col_names=("input_size", "output_size", "num_params"),
        )
    except ImportError:
        print(
            "\n[torchinfo not found]  Install with `pip install torchinfo` "
            "for a detailed layer-by-layer table."
        )


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON/YAML config (defaults to load_config() behaviour)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cuda:0 | cpu | mps | ...",
    )
    main(**vars(parser.parse_args()))
