"""Physical score digital twin: ideal identity + impairment stack → observables.

Pipeline:
    ideal QK^T/sqrt(d)
        → phase / amplitude / angular / crosstalk / detector impairments
        → reconstructed scores
        → metrics (RMS, cosine, top-k, logit SNR proxy)

Ideal layer remains exact. Physical layer is a calibratable error budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any

import torch
import torch.nn.functional as F

from atom.noise import apply_crosstalk


@dataclass
class PhysicalScoreParams:
    """Impairment budget for one score measurement."""

    phase_bias_rad: float = 0.0
    phase_noise_sigma: float = 0.0
    amplitude_error_sigma: float = 0.0
    angular_mix: float = 0.0  # crosstalk strength on key axis
    crosstalk_kernel: int = 3
    detector_noise_sigma: float = 0.0
    adc_bits: int = 0  # 0 = off
    drift: float = 0.0  # additive global offset after reconstruction

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreMetrics:
    rms_error: float
    max_abs_error: float
    cosine: float
    top1_agreement: float
    topk_agreement: float
    logit_snr_db: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScorePrediction:
    """Predicted observables under a physical parameter set."""

    params: PhysicalScoreParams
    metrics: ScoreMetrics
    n_queries: int
    n_keys: int
    trials: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.to_dict(),
            "metrics": self.metrics.to_dict(),
            "n_queries": self.n_queries,
            "n_keys": self.n_keys,
            "trials": self.trials,
        }


def ideal_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Exact digital / binary-phase identity scores."""
    return (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])


def _quantize_levels(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return x
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    n = 2 ** bits
    y = (x - lo) / (hi - lo + 1e-8)
    y = torch.round(y * (n - 1)) / (n - 1)
    return y * (hi - lo) + lo


def physical_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    params: PhysicalScoreParams | None = None,
) -> torch.Tensor:
    """Apply impairment stack to ideal scores (and optionally to Q/K)."""
    params = params or PhysicalScoreParams()
    q_use, k_use = q, k
    if params.amplitude_error_sigma > 0:
        q_use = q * (1.0 + params.amplitude_error_sigma * torch.randn_like(q))
        k_use = k * (1.0 + params.amplitude_error_sigma * torch.randn_like(k))
    if params.phase_bias_rad != 0.0 or params.phase_noise_sigma > 0:
        # Model phase error as rotating the effective real projection slightly
        # via mixing a small orthogonal perturbation into Q (score-domain equivalent).
        noise = params.phase_noise_sigma * torch.randn_like(q_use)
        bias = params.phase_bias_rad
        q_use = q_use * math.cos(bias) - noise * math.sin(bias + 1e-8)
    s = ideal_scores(q_use, k_use)
    if params.angular_mix > 0:
        s = apply_crosstalk(s, strength=params.angular_mix, kernel_size=params.crosstalk_kernel)
    if params.detector_noise_sigma > 0:
        s = s + params.detector_noise_sigma * torch.randn_like(s)
    s = _quantize_levels(s, params.adc_bits)
    if params.drift != 0.0:
        s = s + params.drift
    return s


def score_metrics(
    ideal: torch.Tensor,
    measured: torch.Tensor,
    topk: int = 5,
) -> ScoreMetrics:
    """Compare ideal vs reconstructed score matrices."""
    diff = measured - ideal
    rms = float(diff.pow(2).mean().sqrt())
    max_abs = float(diff.abs().max())
    flat_i = ideal.reshape(-1)
    flat_m = measured.reshape(-1)
    cos = float(F.cosine_similarity(flat_i.unsqueeze(0), flat_m.unsqueeze(0)).item())
    # top-1 / top-k along last dim (keys)
    ti = ideal.argmax(dim=-1)
    tm = measured.argmax(dim=-1)
    top1 = float((ti == tm).float().mean())
    k = min(topk, ideal.shape[-1])
    top_i = ideal.topk(k, dim=-1).indices
    top_m = measured.topk(k, dim=-1).indices
    # agreement: fraction of ideal top-k indices that appear in measured top-k
    agree = 0.0
    n = 0
    for a, b in zip(top_i.reshape(-1, k), top_m.reshape(-1, k)):
        agree += len(set(a.tolist()) & set(b.tolist())) / k
        n += 1
    topk_agr = agree / max(n, 1)
    # logit SNR proxy: signal variance / error variance
    sig = float(ideal.var().clamp_min(1e-12))
    noise = float(diff.var().clamp_min(1e-12))
    snr = 10.0 * math.log10(sig / noise)
    return ScoreMetrics(
        rms_error=rms,
        max_abs_error=max_abs,
        cosine=cos,
        top1_agreement=top1,
        topk_agreement=float(topk_agr),
        logit_snr_db=snr,
    )


def predict_score_observables(
    params: PhysicalScoreParams,
    *,
    n_queries: int = 16,
    n_keys: int = 16,
    head_dim: int = 32,
    trials: int = 8,
    seed: int = 0,
    topk: int = 5,
) -> ScorePrediction:
    """Monte Carlo prediction of measurement metrics under params."""
    g = torch.Generator().manual_seed(seed)
    acc = []
    for t in range(trials):
        q = torch.randn(n_queries, head_dim, generator=g)
        k = torch.randn(n_keys, head_dim, generator=g)
        ideal = ideal_scores(q, k)
        measured = physical_scores(q, k, params)
        acc.append(score_metrics(ideal, measured, topk=topk))
    def mean_metric(name: str) -> float:
        return sum(getattr(m, name) for m in acc) / len(acc)

    metrics = ScoreMetrics(
        rms_error=mean_metric("rms_error"),
        max_abs_error=mean_metric("max_abs_error"),
        cosine=mean_metric("cosine"),
        top1_agreement=mean_metric("top1_agreement"),
        topk_agreement=mean_metric("topk_agreement"),
        logit_snr_db=mean_metric("logit_snr_db"),
    )
    return ScorePrediction(
        params=params,
        metrics=metrics,
        n_queries=n_queries,
        n_keys=n_keys,
        trials=trials,
    )


@dataclass
class AcceptanceGate:
    """Falsifiable thresholds on predicted or measured metrics."""

    max_rms: float = 0.15
    min_cosine: float = 0.95
    min_top1: float = 0.70
    min_topk: float = 0.70
    min_snr_db: float = 5.0

    def evaluate(self, m: ScoreMetrics) -> dict[str, bool]:
        return {
            "rms": m.rms_error <= self.max_rms,
            "cosine": m.cosine >= self.min_cosine,
            "top1": m.top1_agreement >= self.min_top1,
            "topk": m.topk_agreement >= self.min_topk,
            "snr": m.logit_snr_db >= self.min_snr_db,
        }

    def passes(self, m: ScoreMetrics) -> bool:
        return all(self.evaluate(m).values())
