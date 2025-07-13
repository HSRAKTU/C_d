from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


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
        pt_mask:  (B·S, P) — 1 for real points, 0 for padded points.

        Returns
        -------
        (B·S, F) slice embeddings where F = `out_dim`
        """
        feat = self.mlp(pts) * pt_mask.unsqueeze(-1)  # zero padded pts
        masked_val = torch.where(
            pt_mask.bool().unsqueeze(-1),
            feat,
            torch.tensor(float("-inf"), device=feat.device),
        )
        slice_emb = masked_val.max(dim=1).values  # (B·S, F)
        slice_emb[slice_emb == float("-inf")] = 0.0  # empty slice → 0
        return slice_emb


# --------------------------------------------------------------------------- #
#  Temporal encoder (bi-LSTM)                                                 #
# --------------------------------------------------------------------------- #
class LSTMEncoder(nn.Module):
    """Bidirectional LSTM over the (unpadded) slice sequence."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        seq: (B, S, F) — slice embeddings.

        Returns
        -------
        (B, 2·hidden) — concatenated [fwd, bwd] last hidden states.
        """
        _, (h_n, _) = self.lstm(seq)  # h_n: (2, B, H)
        h_fwd, h_bwd = h_n[-2], h_n[-1]  # (B, H) each
        return torch.cat([h_fwd, h_bwd], dim=-1)


# --------------------------------------------------------------------------- #
#  Full Cd regressor                                                          #
# --------------------------------------------------------------------------- #
class CdRegressor(nn.Module):
    """
    Notebook-style wrapper: `forward(slices, point_mask)`.
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
    def forward(self, x) -> torch.Tensor:
        """
        Args
        ----
        x: 2-tuple `(slices, point_mask)`
            slices      – (B, S, P, 2)
            point_mask  – (B, S, P)

        Returns
        -------
        (B, 1) — Cd prediction
        """
        slices, point_mask = x
        B, S, P, _ = slices.shape

        # PointNet per slice -------------------------------------------------
        flat_pts = slices.view(B * S, P, 2)
        flat_mask = point_mask.view(B * S, P)
        slice_emb = self.slice_encoder(flat_pts, flat_mask)  # (B·S, F)
        slice_emb = slice_emb.view(B, S, -1)  # (B, S, F)

        # LSTM over slice sequence ------------------------------------------
        seq_feat = self.temporal(slice_emb)  # (B, 2H)
        return self.head(seq_feat).squeeze(-1)  # (B,)

    # ----------------------- convenience for export --------------------- #
    @staticmethod
    def example_input(
        batch_size: int = 2, S: int = 80, P: int = 6500
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dummy input for tracing / ONNX export.
        Returns a tuple `(slices, point_mask)`.
        """
        slices = torch.randn(batch_size, S, P, 2)
        p_mask = torch.ones(batch_size, S, P)
        return (slices, p_mask)
