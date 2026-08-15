"""Uncertainty-bearing parameters with provenance.

Physical quantities in ATOM (M#, n_erase, throughput, ...) are not bare
floats. Each value carries a source, an optional interval, and notes so
plans and claim audits can report calibration state.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

Source = Literal["algebra", "literature", "measured", "assumed", "placeholder"]


@dataclass(frozen=True)
class UncertainParam:
    """Scalar parameter with traceability metadata."""

    name: str
    value: float
    source: Source
    unit: str = ""
    lower: float | None = None
    upper: float | None = None
    geometry: str = ""
    notes: str = ""

    def interval(self) -> tuple[float | None, float | None]:
        return self.lower, self.upper

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Defaults aligned with capacity.py / refresh.py literature-class figures.
M_NUMBER = UncertainParam(
    name="m_number",
    value=2.0,
    source="literature",
    unit="cm^-1_class",
    lower=1.0,
    upper=5.0,
    geometry="Fe:LiNbO3 90-degree class",
    notes="Conservative literature-class placeholder; not a local lab measurement.",
)

N_ERASE = UncertainParam(
    name="n_erase",
    value=500.0,
    source="placeholder",
    unit="readouts",
    lower=100.0,
    upper=1e8,
    geometry="unfixed single-color continuous readout vs two-color regimes",
    notes="Regime-dependent. Single-color continuous read is short; two-color/fixing can be orders of magnitude longer.",
)

ETA_MIN = UncertainParam(
    name="eta_min",
    value=1e-4,
    source="assumed",
    unit="diffraction_efficiency",
    lower=1e-5,
    upper=1e-3,
    notes="Minimum usable diffraction efficiency for capacity counts.",
)


def param_registry() -> dict[str, UncertainParam]:
    return {
        M_NUMBER.name: M_NUMBER,
        N_ERASE.name: N_ERASE,
        ETA_MIN.name: ETA_MIN,
    }
