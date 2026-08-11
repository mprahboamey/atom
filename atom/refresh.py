"""Readout erase budget and rack-scale refresh scheduling (stub).

Photorefractive holograms fade under same-wavelength readout. This module
does not implement optical writing. It estimates when stored weights are
expected to fall below a usable diffraction-efficiency floor, and schedules
reinjection of digital masters into crystal banks — the DRAM-refresh
analogue for a holographic weight volume.

Material defaults are conservative Fe:LiNbO3-class order-of-magnitude
placeholders. Replace with measured erase rates before using as a hard
operations limit.

Writing / rewriting the crystal is a hardware problem (beam control,
exposure schedule, partial-erase of neighbors). The rack architecture
below assumes rewrite is optical-in-place, not a mechanically jittery
swapped cartridge on every refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MaterialEraseParams:
    """Erase-under-read parameters for one photorefractive medium.

    Model (first-order):

        eta(n) ≈ eta_0 * exp(-n / n_erase)

    where n is the number of full-volume (or full-bank) coherent readouts
    at a reference intensity. Real media depend on intensity, wavelength,
    doping, temperature, and fixing state; this is a planning curve only.

    Defaults: iron-doped lithium niobate *class* of order-of-magnitude
    figures for *unfixed* gratings under continuous readout. Fixed or
    two-color schemes can raise n_erase by orders of magnitude — update
    when you have bench numbers.
    """

    # Diffraction efficiency right after a fresh write (fraction).
    eta0: float = 1e-2

    # Characteristic readout count where eta falls by 1/e at reference intensity.
    n_erase: float = 500.0

    # Minimum eta still usable for the score path at target SNR.
    eta_min: float = 1e-4

    # Reference read intensity used when quoting n_erase (mW/cm^2).
    i_ref_mw_cm2: float = 10.0

    # Optional intensity scaling: n_erase_eff ≈ n_erase * (i_ref / i_actual).
    # Higher intensity erases faster (fewer reads to the same fade).
    intensity_linear: bool = True

    material_name: str = "Fe:LiNbO3 (unfixed, placeholder)"

    def reads_to_eta_min(self, i_mw_cm2: float | None = None) -> float:
        """Expected read count until eta hits eta_min (same exposure model)."""
        if self.eta0 <= 0 or self.eta_min <= 0:
            raise ValueError("eta0 and eta_min must be positive")
        if self.eta_min > self.eta0:
            return 0.0
        n = self.n_erase * math.log(self.eta0 / self.eta_min)
        if i_mw_cm2 is not None and self.intensity_linear and i_mw_cm2 > 0:
            n = n * (self.i_ref_mw_cm2 / i_mw_cm2)
        return n

    def eta_after_reads(self, n_reads: float, i_mw_cm2: float | None = None) -> float:
        """eta after n_reads under the exponential fade model."""
        n_char = self.n_erase
        if i_mw_cm2 is not None and self.intensity_linear and i_mw_cm2 > 0:
            n_char = self.n_erase * (self.i_ref_mw_cm2 / i_mw_cm2)
        if n_char <= 0:
            return 0.0
        return self.eta0 * math.exp(-n_reads / n_char)


@dataclass
class CrystalBank:
    """One addressable holographic bank in a rack (logical, not physical driver)."""

    bank_id: int
    material: MaterialEraseParams = field(default_factory=MaterialEraseParams)
    read_count: float = 0.0
    # Fraction of channels rewritten on last refresh (1.0 = full bank).
    last_rewrite_fraction: float = 1.0
    offline: bool = False

    def record_reads(self, n: float = 1.0) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.read_count += n

    def eta_now(self, i_mw_cm2: float | None = None) -> float:
        return self.material.eta_after_reads(self.read_count, i_mw_cm2)

    def reads_remaining(self, i_mw_cm2: float | None = None) -> float:
        budget = self.material.reads_to_eta_min(i_mw_cm2)
        return max(0.0, budget - self.read_count)

    def needs_refresh(self, safety_margin: float = 0.2, i_mw_cm2: float | None = None) -> bool:
        """True if remaining budget is below safety_margin * full budget."""
        budget = self.material.reads_to_eta_min(i_mw_cm2)
        if budget <= 0:
            return True
        return self.reads_remaining(i_mw_cm2) <= safety_margin * budget

    def mark_rewritten(self, fraction: float = 1.0) -> None:
        """Call after a successful hardware rewrite. Resets erase clock for fraction.

        Full rewrite (fraction=1) zeros read_count. Partial rewrite scales
        read_count down — a coarse model until channel-level counters exist.
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be in (0, 1]")
        self.read_count *= 1.0 - fraction
        self.last_rewrite_fraction = fraction
        self.offline = False


@dataclass
class RackRefreshPolicy:
    """Schedule reinjection across banks without assuming mechanical swaps.

    Design intent:
    - Digital host holds the weight master (or a compressed master).
    - Each bank is a crystal (or sub-volume) with its own read counter.
    - Refresh is *optical rewrite in place* from the host, not a robot
      reseating modules every N reads — mechanical jitter is a tax we
      explicitly want to avoid in the rack story.
    - Stripe traffic so one bank can go offline for rewrite while others serve.

    Unsolved (hardware):
    - Beam addressing, exposure time, closed-loop eta sensing
    - Crosstalk during write to neighboring angular channels
    - Power and thermal limits of continuous rewrite
    """

    banks: list[CrystalBank]
    safety_margin: float = 0.2
    i_mw_cm2: float | None = None

    def banks_needing_refresh(self) -> list[CrystalBank]:
        return [b for b in self.banks if b.needs_refresh(self.safety_margin, self.i_mw_cm2)]

    def record_read_on_bank(self, bank_id: int, n: float = 1.0) -> None:
        for b in self.banks:
            if b.bank_id == bank_id:
                b.record_reads(n)
                return
        raise KeyError(f"bank_id {bank_id} not found")

    def plan_refresh(self) -> list[dict]:
        """Return a list of refresh jobs (software plan only).

        Each job is a dict the control plane could hand to a future write
        driver: which bank, estimated eta, suggested full vs partial.
        """
        jobs = []
        for b in self.banks_needing_refresh():
            jobs.append(
                {
                    "bank_id": b.bank_id,
                    "read_count": b.read_count,
                    "eta_est": b.eta_now(self.i_mw_cm2),
                    "reads_remaining": b.reads_remaining(self.i_mw_cm2),
                    "action": "optical_rewrite_in_place",
                    "suggested_fraction": 1.0,
                    "note": "Hardware write path not implemented; plan only.",
                }
            )
        return jobs

    def summary(self) -> dict:
        return {
            "n_banks": len(self.banks),
            "needing_refresh": [b.bank_id for b in self.banks_needing_refresh()],
            "eta": {b.bank_id: b.eta_now(self.i_mw_cm2) for b in self.banks},
            "reads_remaining": {
                b.bank_id: b.reads_remaining(self.i_mw_cm2) for b in self.banks
            },
        }


def default_fe_linbo3_rack(n_banks: int = 8) -> RackRefreshPolicy:
    """Convenience: n_banks with shared placeholder Fe:LiNbO3 erase curve."""
    mat = MaterialEraseParams()
    banks = [CrystalBank(bank_id=i, material=mat) for i in range(n_banks)]
    return RackRefreshPolicy(banks=banks)
