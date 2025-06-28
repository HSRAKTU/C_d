"""
CdDataset
~~~~~~~~~
Loads padded & masked 2-D slices from *.npz files, (optionally) normalises
the C_d StandardScaler, and returns tensors ready for
PyTorch-Ignite loops.

Normalisation rules
-------------------
* Fit the scaler **once** on the *train* split (all points where point_mask==1).
* Persist it to `constants.SCALER_FILE`.
* All other splits **load** the persisted scaler – never refit.
* Cd (target) is never scaled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.config.constants import SCALER_FILE
from src.utils.io import load_design_ids, load_scaler, save_scaler
from src.utils.logger import logger


class CdDataset(Dataset):
    """
    Dataset yielding (x = (slices (S,P,2), point_mask (S,P), slice_mask (S,)), y = target(scalar)) tensors.

    Args:
        root_dir:   directory containing the padded/masked *.npz files.
        split:      "train" | "val" | "test".
        fit_scaler: if True, fit a new StandardScaler and persist it.
                    Only set True on the *train* split of the FIRST run.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        fit_scaler: bool = False,
    ):
        self.root_dir = Path(root_dir)
        design_ids = load_design_ids(split)
        all_npz = self.root_dir.glob("*_axis-*.npz")
        self.files = sorted(
            p for p in all_npz if p.stem.split("_axis-")[0] in design_ids
        )
        if not self.files:
            raise RuntimeError(f"No data found in {self.root_dir}")

        # ----------------------------------------------------------------- #
        # 1) Build / load Cd-scaler                                            #
        # ----------------------------------------------------------------- #
        try:
            if fit_scaler:
                self.scaler = self._fit_and_save_scaler()
            else:
                self.scaler = load_scaler(SCALER_FILE)
            if not self.scaler.mean_ or not self.scaler.scale_:
                raise ValueError(
                    "Transformation Scaler not loader correctly. Can't get the mean and scale."
                )
        except Exception as e:
            logger.error(f"Error in getting transformation scaler. Error msg: {e}")
            raise e

        logger.info(
            f"{split:<5} dataset with {len(self)} samples | "
            f"Scaler mean={self.scaler.mean_[0].round(3)} "
            f"scale={self.scaler.scale_[0].round(3)}"
        )

    # ------------------------------ scaler ------------------------------ #
    def _fit_and_save_scaler(self) -> StandardScaler:
        """
        Fit a StandardScalar on Cd values from the split.
        """
        logger.info("Fitting StadnardScalar on Cd targets")
        cds = np.array(
            [self._load_npz(fp, raw=True)[3] for fp in self.files], dtype=np.float32
        ).reshape(-1, 1)  # (N, 1)
        scaler = StandardScaler().fit(cds)
        save_scaler(scaler, SCALER_FILE)
        logger.info(f"Scaler saved → {SCALER_FILE}")
        return scaler

    # ---------------------------- I/O utils ----------------------------- #
    def _load_npz(self, fp: Path, raw: bool = False):
        """Return (slices, point_mask, slice_mask, Cd)"""
        data = np.load(fp, allow_pickle=True)
        slices = data["slices"]  # (S, P, 2)
        p_mask = data["point_mask"]  # (S, P)
        s_mask = data["slice_mask"]  # (S,)
        if "Cd" not in data:
            raise KeyError(
                f"{fp} has no 'Cd' field – regenerate with updated preprocessing."
            )
        cd = data["Cd"]
        if raw:
            return slices, p_mask, s_mask, cd
        # ── scale Cd only ──
        cd_scaled = float(
            self.scaler.transform(np.array(cd, dtype=np.float32).reshape(-1, 1))[0, 0]
        )
        return slices, p_mask, s_mask, cd_scaled

    # ------------------------- standard Dataset ------------------------- #
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        """
        Returns (x,y) where x is a 3-tuple of tensors and y is of shape (1,)
        """
        slices, p_mask, s_mask, cd = self._load_npz(self.files[idx])
        x = (
            torch.from_numpy(slices).float(),
            torch.from_numpy(p_mask).float(),
            torch.from_numpy(s_mask).float(),
        )
        y = torch.tensor(cd, dtype=torch.float32)
        return x, y
