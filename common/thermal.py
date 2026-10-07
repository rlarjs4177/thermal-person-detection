from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


THERMAL_MODES = ("heatcore", "heatcore_diffusion", "full")


class ThermalModule(nn.Module):
    """Paper equations (1)-(8), without detector-specific normalization."""

    def __init__(self, mode: str = "full", kernel_size: int = 7, epsilon: float = 1e-6):
        super().__init__()
        if mode not in THERMAL_MODES:
            raise ValueError(f"thermal mode must be one of {THERMAL_MODES}, got {mode!r}")
        if kernel_size != 7:
            raise ValueError("The paper fixes the local-mean kernel to 7x7.")
        self.mode, self.kernel_size, self.epsilon = mode, kernel_size, epsilon
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0)) if mode != "heatcore" else None
        self.gamma = nn.Parameter(torch.tensor(1.0)) if mode == "full" else None
        self.last_debug: dict[str, Tensor | str] = {}

    def forward(self, image_01: Tensor) -> Tensor:
        if image_01.ndim != 4 or image_01.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W], got {tuple(image_01.shape)}")
        if not image_01.is_floating_point():
            raise TypeError("ThermalModule requires a floating point tensor in [0,1].")

        # Eq. (1)
        x = image_01.mean(dim=1, keepdim=True)
        # Eq. (2): deterministic replicate padding keeps the output resolution.
        radius = self.kernel_size // 2
        local_mean = F.avg_pool2d(F.pad(x, (radius,) * 4, mode="replicate"), self.kernel_size, stride=1)
        # Eq. (3)
        heat_core = F.relu(x - local_mean)
        # Eqs. (4)-(5): forward differences, replicate at right/bottom boundary.
        gx = F.pad(x, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:] - x
        gy = F.pad(x, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :] - x
        # Eqs. (6)-(7)
        diffusion = torch.sqrt(gx.square() + gy.square() + self.epsilon)
        decay = torch.exp(-diffusion)
        # Eq. (8): inactive terms are structurally omitted, not multiplied by zero.
        logits = self.alpha * heat_core
        if self.beta is not None:
            logits = logits + self.beta * diffusion
        if self.gamma is not None:
            logits = logits + self.gamma * decay
        weight = torch.sigmoid(logits)
        self.last_debug = {
            "mode": self.mode,
            "X": x,
            "local_mean": local_mean,
            "H": heat_core,
            "Gx": gx,
            "Gy": gy,
            "D": diffusion,
            "R": decay,
            "W": weight,
        }
        return weight

    def learned_values(self) -> dict[str, float | None]:
        return {
            "alpha_w1": float(self.alpha.detach().cpu()),
            "beta_w2": None if self.beta is None else float(self.beta.detach().cpu()),
            "gamma_w3": None if self.gamma is None else float(self.gamma.detach().cpu()),
        }


def enhance_feature(feature: Tensor, weight: Tensor) -> Tensor:
    """Paper Eq. (9): F' = F * (1 + resize(W))."""
    aligned = F.interpolate(weight, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    return feature * (1.0 + aligned)


def sample_weight_at_locations(weight: Tensor, sampling_locations: Tensor) -> Tensor:
    """Differentiably sample W at deformable-attention [0,1] (x,y) locations."""
    if sampling_locations.ndim != 6:
        raise ValueError(f"expected [B,Q,heads,levels,points,2], got {tuple(sampling_locations.shape)}")
    b, q, heads, levels, points, _ = sampling_locations.shape
    grid = sampling_locations.mul(2.0).sub(1.0).reshape(b, q * heads, levels * points, 2)
    sampled = F.grid_sample(weight, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled[:, 0].reshape(b, q, heads, levels, points)

