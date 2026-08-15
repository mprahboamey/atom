"""Machine-readable physical claims and audit report.

Each claim binds: quantity, ideal equation, physical variables,
prediction, acceptance gate, evidence status, required bench measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from atom.physical_score import (
    AcceptanceGate,
    PhysicalScoreParams,
    predict_score_observables,
)
from atom.uncertainty import M_NUMBER, N_ERASE, ETA_MIN, param_registry


class ClaimStatus(str, Enum):
    EXACT = "EXACT"
    PREDICTED = "PREDICTED"
    ASSUMPTION_BOUNDED = "ASSUMPTION_BOUNDED"
    REGIME_DEPENDENT = "REGIME_DEPENDENT"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


@dataclass
class Claim:
    id: str
    title: str
    quantity: str
    ideal_equation: str
    status: ClaimStatus
    physical_variables: list[str] = field(default_factory=list)
    evidence: str = ""
    predicted_observables: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, Any] = field(default_factory=dict)
    required_measurement: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def build_claims(
    score_params: PhysicalScoreParams | None = None,
    gate: AcceptanceGate | None = None,
) -> list[Claim]:
    score_params = score_params or PhysicalScoreParams(
        phase_noise_sigma=0.05,
        amplitude_error_sigma=0.03,
        angular_mix=0.05,
        detector_noise_sigma=0.02,
        adc_bits=8,
    )
    gate = gate or AcceptanceGate()
    pred = predict_score_observables(score_params, trials=6)
    gate_eval = gate.evaluate(pred.metrics)

    claims = [
        Claim(
            id="CLAIM_001",
            title="Binary-phase score identity",
            quantity="QK^T / sqrt(d)",
            ideal_equation="binary-phase interference product equals scaled dot-product for real Q,K",
            status=ClaimStatus.EXACT,
            physical_variables=[],
            evidence="algebra + unit tests + hybrid checkpoint parity",
            required_measurement="none for ideal layer",
            predicted_observables={"max_abs_error": 0.0},
            acceptance={"max_abs_error": 0.0},
        ),
        Claim(
            id="CLAIM_002",
            title="Optical score fidelity",
            quantity="QK^T / sqrt(d)",
            ideal_equation="binary-phase interference",
            status=ClaimStatus.PREDICTED,
            physical_variables=[
                "phase_bias",
                "phase_noise",
                "amplitude_error",
                "angular_crosstalk",
                "detector_noise",
                "adc_bits",
                "drift",
            ],
            evidence="digital twin Monte Carlo under PhysicalScoreParams",
            predicted_observables=pred.metrics.to_dict(),
            acceptance={
                "gates": gate_eval,
                "thresholds": asdict(gate),
                "passes_prediction": gate.passes(pred.metrics),
            },
            required_measurement="interference / score matrix on fixed Q,K with logged optical settings",
            parameters=score_params.to_dict(),
        ),
        Claim(
            id="CLAIM_003",
            title="Holographic capacity",
            quantity="usable multiplexed weight slots",
            ideal_equation="eta ≈ (M#/M)^2; usable channels ≈ M#/sqrt(eta_min)",
            status=ClaimStatus.ASSUMPTION_BOUNDED,
            physical_variables=["m_number", "eta_min", "geometry"],
            evidence="capacity.py + literature-class M#",
            parameters={
                "m_number": M_NUMBER.to_dict(),
                "eta_min": ETA_MIN.to_dict(),
            },
            required_measurement="multiplexed diffraction efficiency vs hologram count",
        ),
        Claim(
            id="CLAIM_004",
            title="Read durability",
            quantity="eta(n) under repeated readout",
            ideal_equation="eta(n) ≈ eta0 * exp(-n/n_erase) in simple unfixed model",
            status=ClaimStatus.REGIME_DEPENDENT,
            physical_variables=["n_erase", "recording_regime", "wavelength", "fluence"],
            evidence="refresh.py exponential model; n_erase uncalibrated locally",
            parameters={"n_erase": N_ERASE.to_dict()},
            required_measurement="eta vs read fluence / readout count for chosen recording regime",
        ),
        Claim(
            id="CLAIM_005",
            title="Optical acceleration",
            quantity="wall-time or energy per score matmul",
            ideal_equation="throughput model in scale_plan.ThroughputAssumptions",
            status=ClaimStatus.NOT_ESTABLISHED,
            physical_variables=[
                "modulation_latency",
                "optical_path",
                "detector_latency",
                "ADC",
                "digital_softmax",
                "sync",
            ],
            evidence="assumption slots only; no board or bench joules in repo",
            required_measurement="end-to-end latency and energy for score path vs digital baseline",
        ),
    ]
    return claims


def format_audit(claims: list[Claim] | None = None) -> str:
    claims = claims or build_claims()
    lines = [
        "ATOM PHYSICAL CLAIM AUDIT",
        "=========================",
        "",
    ]
    for c in claims:
        lines.append(f"{c.id}")
        lines.append(f"  {c.title}")
        lines.append(f"  Status: {c.status.value}")
        lines.append(f"  Quantity: {c.quantity}")
        lines.append(f"  Ideal: {c.ideal_equation}")
        if c.physical_variables:
            lines.append(f"  Physical: {', '.join(c.physical_variables)}")
        if c.evidence:
            lines.append(f"  Evidence: {c.evidence}")
        if c.predicted_observables:
            lines.append(f"  Predicted: {c.predicted_observables}")
        if c.acceptance:
            lines.append(f"  Acceptance: {c.acceptance}")
        if c.required_measurement:
            lines.append(f"  Required measurement: {c.required_measurement}")
        if c.parameters:
            lines.append(f"  Parameters: {c.parameters}")
        lines.append("")
    lines.append("Uncertainty registry:")
    for name, p in param_registry().items():
        lines.append(
            f"  {name}={p.value} source={p.source} interval=[{p.lower}, {p.upper}] {p.notes[:60]}"
        )
    return "\n".join(lines)
