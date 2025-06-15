"""
Cd-Regressor network used in the original notebook.

Pipeline
--------
(B, S, P, 2)  →  PointNet2D  →  (B, S, F)
              ▸ max-pool over P (masked)
(B, S, F)     →  Bi-LSTM     →  (B, 2·H)
(B, 2·H)      →  MLP head    →  (B, 1)   ← Cd prediction

Mask support
------------
* `point_mask` (B, S, P) filters padded points before the max-pool.
* `slice_mask` (B, S)   is converted to sequence lengths so the LSTM
  only processes real slices (via `pack_padded_sequence`).

Nothing is *inverse-scaled* here: only the input coordinates are scaled
(by the dataset) and Cd itself is predicted in physical units.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# --------------------------------------------------------------------------- #
#  PointNet-style slice encoder                                               #
# --------------------------------------------------------------------------- #
class PointNet2D(nn.Module):
    """Per-point MLP → masked global-max → slice embedding."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pts: torch.Tensor, pt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        pts:      (B·S, P, 2)
        pt_mask:  (B·S, P)

        Returns
        -------
        slice_emb: (B·S, out_dim)
        """
        feat = self.mlp(pts)  # (B·S, P, F)
        feat = feat * pt_mask.unsqueeze(-1)  # zero padded points
        # Set masked values to -inf so max ignores them, then replace -inf→0
        masked_val = torch.where(
            pt_mask.bool().unsqueeze(-1),
            feat,
            torch.tensor(float("-inf"), device=feat.device),
        )
        slice_emb = masked_val.max(dim=1).values  # (B·S, F)
        slice_emb[slice_emb == float("-inf")] = 0.0  # empty slice → zeros
        return slice_emb


# --------------------------------------------------------------------------- #
#  Temporal encoder (bi-LSTM)                                                 #
# --------------------------------------------------------------------------- #
class LSTMEncoder(nn.Module):
    """Bidirectional LSTM that ignores padded slices via packing."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, seq: torch.Tensor, slice_mask: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        seq:         (B, S, F)
        slice_mask:  (B, S)

        Returns
        -------
        (B, 2·hidden)  concatenated last-layer [fwd, bwd] hidden states
        """
        lengths = slice_mask.sum(dim=1).clamp(min=1).long()  # avoid 0-len
        packed = pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (2, B, H)
        # last layer → index -2 (fwd) & -1 (bwd)
        h_fwd, h_bwd = h_n[-2], h_n[-1]  # (B, H) each
        return torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2H)


# --------------------------------------------------------------------------- #
#  Full regressor                                                             #
# --------------------------------------------------------------------------- #
class CdRegressor(nn.Module):
    """
    Wrapper that matches the notebook-style `forward(slices, p_mask, s_mask)`.
    """

    def __init__(
        self,
        point_feat: int = 128,
        lstm_hidden: int = 128,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.slice_encoder = PointNet2D(out_dim=point_feat)
        self.temporal = LSTMEncoder(in_dim=point_feat, hidden=lstm_hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )

    # -------------------------- public interface ------------------------ #
    def forward(
        self,
        slices: torch.Tensor,
        point_mask: torch.Tensor,
        slice_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
        ----
        slices:      (B, S, P, 2)
        point_mask:  (B, S, P)
        slice_mask:  (B, S)

        Returns
        -------
        Cd: (B, 1)
        """
        B, S, P, _ = slices.shape
        # Flatten batch & slice dims for PointNet
        flat_pts = slices.view(B * S, P, 2)
        flat_mask = point_mask.view(B * S, P)
        slice_embed = self.slice_encoder(flat_pts, flat_mask)  # (B·S, F)
        slice_embed = slice_embed.view(B, S, -1)  # (B, S, F)

        # Mask out padded slices (zero vectors before packing)
        slice_embed = slice_embed * slice_mask.unsqueeze(-1)

        seq_feat = self.temporal(slice_embed, slice_mask)  # (B, 2H)
        return self.head(seq_feat)  # (B, 1)

    # ----------------------- convenience for export --------------------- #
    @staticmethod
    def example_input(
        batch_size: int = 2, S: int = 80, P: int = 6500
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a dummy input triple for tracing / ONNX export."""
        slices = torch.randn(batch_size, S, P, 2)
        p_mask = torch.ones(batch_size, S, P)
        s_mask = torch.ones(batch_size, S)
        return slices, p_mask, s_mask
