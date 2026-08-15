"""Inverse calibration: observed scores → estimated physical parameters.

Fits a subset of PhysicalScoreParams so physical_scores(q,k,params)
matches an observed interference / score matrix. Works on synthetic
observations first; the same API takes bench matrices later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

from atom.physical_score import (
    PhysicalScoreParams,
    ideal_scores,
    physical_scores,
    score_metrics,
)


@dataclass
class CalibrationResult:
    params: PhysicalScoreParams
    loss: float
    metrics: dict[str, float]
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.to_dict(),
            "loss": self.loss,
            "metrics": self.metrics,
            "steps": self.steps,
        }


def fit_physical_params(
    q: torch.Tensor,
    k: torch.Tensor,
    observed: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 0.05,
    fit_phase_noise: bool = True,
    fit_amplitude: bool = True,
    fit_crosstalk: bool = True,
    fit_detector: bool = True,
    fit_drift: bool = True,
) -> CalibrationResult:
    """Gradient-free coordinate / Adam fit on unconstrained raw parameters."""
    # Parameterize positive quantities via softplus-like exp
    raw = {
        "phase_noise": torch.tensor(0.05, requires_grad=True),
        "amplitude": torch.tensor(0.05, requires_grad=True),
        "crosstalk": torch.tensor(0.05, requires_grad=True),
        "detector": torch.tensor(0.05, requires_grad=True),
        "drift": torch.tensor(0.0, requires_grad=True),
        "phase_bias": torch.tensor(0.0, requires_grad=True),
    }
    opt = torch.optim.Adam(list(raw.values()), lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        opt.zero_grad()
        params = PhysicalScoreParams(
            phase_bias_rad=float(raw["phase_bias"].item()) if not raw["phase_bias"].requires_grad else raw["phase_bias"],
            phase_noise_sigma=torch.nn.functional.softplus(raw["phase_noise"]) if fit_phase_noise else 0.0,
            amplitude_error_sigma=torch.nn.functional.softplus(raw["amplitude"]) if fit_amplitude else 0.0,
            angular_mix=torch.sigmoid(raw["crosstalk"]) if fit_crosstalk else 0.0,
            detector_noise_sigma=torch.nn.functional.softplus(raw["detector"]) if fit_detector else 0.0,
            drift=raw["drift"] if fit_drift else 0.0,
        )
        # physical_scores uses random draws; for fit use deterministic reconstruction
        pred = _deterministic_physical(q, k, params, raw)
        loss = torch.mean((pred - observed) ** 2)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())

    final = PhysicalScoreParams(
        phase_bias_rad=float(raw["phase_bias"].detach()),
        phase_noise_sigma=float(torch.nn.functional.softplus(raw["phase_noise"]).detach()) if fit_phase_noise else 0.0,
        amplitude_error_sigma=float(torch.nn.functional.softplus(raw["amplitude"]).detach()) if fit_amplitude else 0.0,
        angular_mix=float(torch.sigmoid(raw["crosstalk"]).detach()) if fit_crosstalk else 0.0,
        detector_noise_sigma=float(torch.nn.functional.softplus(raw["detector"]).detach()) if fit_detector else 0.0,
        drift=float(raw["drift"].detach()) if fit_drift else 0.0,
    )
    with torch.no_grad():
        pred = physical_scores(q, k, final)
        m = score_metrics(ideal_scores(q, k), observed)
    return CalibrationResult(
        params=final,
        loss=last_loss,
        metrics=m.to_dict(),
        steps=steps,
    )


def _deterministic_physical(
    q: torch.Tensor,
    k: torch.Tensor,
    params: PhysicalScoreParams,
    raw: dict,
) -> torch.Tensor:
    """Differentiable proxy: amplitude scale + crosstalk + noise-free detector gain + drift."""
    # Use expected-value style: scale Q/K by (1+eps) mean via learned amplitude as gain mismatch
    amp = torch.nn.functional.softplus(raw["amplitude"])
    q2 = q * (1.0 + amp)
    k2 = k * (1.0 + amp)
    s = ideal_scores(q2, k2)
    mix = torch.sigmoid(raw["crosstalk"])
    # differentiable box mix on last dim
    if s.shape[-1] >= 3:
        pad = F_pad = torch.nn.functional.pad
        x = s.unsqueeze(1)
        # depthwise avg as crosstalk proxy
        ker = torch.ones(1, 1, 1, 3, device=s.device, dtype=s.dtype) / 3.0
        # reshape to (1,1,Q,K)
        if s.dim() == 2:
            x = s.unsqueeze(0).unsqueeze(0)
            y = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (1, 1, 0, 0)), ker)
            mixed = y.squeeze(0).squeeze(0)
        else:
            mixed = s
        s = (1.0 - mix) * s + mix * mixed
    s = s + raw["drift"]
    return s


def synthetic_calibration_demo(
    true: PhysicalScoreParams | None = None,
    n: int = 24,
    head_dim: int = 32,
    seed: int = 1,
) -> CalibrationResult:
    """Generate observed scores from known params and recover an estimate."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, head_dim, generator=g)
    k = torch.randn(n, head_dim, generator=g)
    true = true or PhysicalScoreParams(
        phase_noise_sigma=0.08,
        amplitude_error_sigma=0.05,
        angular_mix=0.1,
        detector_noise_sigma=0.02,
        drift=0.01,
    )
    with torch.no_grad():
        observed = physical_scores(q, k, true)
    return fit_physical_params(q, k, observed, steps=150)
