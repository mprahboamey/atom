"""Tests for claim audit, physical score twin, calibration."""

from __future__ import annotations

import math

import torch

from atom.physical_score import (
    ideal_scores,
    physical_scores,
    PhysicalScoreParams,
    score_metrics,
    predict_score_observables,
    AcceptanceGate,
)
from atom.claims import build_claims, ClaimStatus
from atom.calibrate import synthetic_calibration_demo


def test_ideal_identity():
    q = torch.randn(8, 16)
    k = torch.randn(8, 16)
    s = ideal_scores(q, k)
    d = (q @ k.T) / math.sqrt(16)
    assert torch.allclose(s, d)


def test_physical_zero_impairment_matches_ideal():
    q = torch.randn(8, 16)
    k = torch.randn(8, 16)
    s0 = ideal_scores(q, k)
    s1 = physical_scores(q, k, PhysicalScoreParams())
    assert torch.allclose(s0, s1)


def test_prediction_runs():
    pred = predict_score_observables(
        PhysicalScoreParams(phase_noise_sigma=0.1, detector_noise_sigma=0.05),
        trials=3,
    )
    assert pred.metrics.rms_error >= 0
    assert -1.0 <= pred.metrics.cosine <= 1.0


def test_gate():
    pred = predict_score_observables(PhysicalScoreParams(), trials=2)
    assert AcceptanceGate(max_rms=1e6, min_cosine=0.0, min_top1=0.0, min_topk=0.0, min_snr_db=-100).passes(
        pred.metrics
    )


def test_claims_contain_exact_and_predicted():
    claims = build_claims()
    statuses = {c.status for c in claims}
    assert ClaimStatus.EXACT in statuses
    assert ClaimStatus.PREDICTED in statuses
    assert any(c.id == "CLAIM_001" for c in claims)


def test_calibration_demo_runs():
    cal = synthetic_calibration_demo()
    assert cal.steps > 0
    assert cal.loss >= 0
