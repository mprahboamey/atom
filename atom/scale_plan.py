"""Predictable scale-out planner for hybrid MoE training targets.

Maps total parameter goals (2T, 20T, 200T, ...) onto:
  - MoE shape (experts, top-k, active params)
  - token budget (Chinchilla-style active-param rule)
  - crystal/rack banks from usable M# capacity
  - digital vs optical work split (scores optical, FFN experts digital/banks)
  - coarse FLOP and calendar estimates under stated throughput assumptions

This does not train a 20T model. It makes requirements for 2T and 200T
the same *kind* of plan so scale-out is arithmetic, not a new invention
at each order of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

from atom.capacity import CapacityParams, usable_capacity


@dataclass(frozen=True)
class MoEShape:
    """Sparse MoE layout for a target total parameter count."""

    total_params: float
    n_experts: int
    top_k: int
    active_frac: float
    active_params: float
    dense_equivalent_params: float  # ~active path size

    @staticmethod
    def from_total(
        total_params: float,
        n_experts: int = 64,
        top_k: int = 2,
        expert_param_frac: float = 0.85,
    ) -> "MoEShape":
        """Build shape assuming most params live in experts.

        active_params ≈ (1 - expert_param_frac) * total
                        + expert_param_frac * total * (top_k / n_experts)
        """
        if n_experts < 1 or top_k < 1 or top_k > n_experts:
            raise ValueError("need 1 <= top_k <= n_experts")
        if not 0.0 < expert_param_frac <= 1.0:
            raise ValueError("expert_param_frac in (0, 1]")
        shared = (1.0 - expert_param_frac) * total_params
        expert_pool = expert_param_frac * total_params
        active = shared + expert_pool * (top_k / n_experts)
        return MoEShape(
            total_params=float(total_params),
            n_experts=int(n_experts),
            top_k=int(top_k),
            active_frac=float(active / total_params),
            active_params=float(active),
            dense_equivalent_params=float(active),
        )


@dataclass(frozen=True)
class TokenBudget:
    """Training token targets from active parameter count.

    tokens = multiplier * active_params
    Default multiplier=20 is a Chinchilla-class rule of thumb for dense
    models; MoE often uses more tokens relative to *active* size. Expose
    multipliers rather than hide them.
    """

    active_params: float
    multiplier: float
    tokens: float

    @staticmethod
    def from_active(active_params: float, multiplier: float = 20.0) -> "TokenBudget":
        if active_params <= 0 or multiplier <= 0:
            raise ValueError("active_params and multiplier must be positive")
        return TokenBudget(
            active_params=float(active_params),
            multiplier=float(multiplier),
            tokens=float(multiplier * active_params),
        )


@dataclass(frozen=True)
class StoragePlan:
    """How many usable-capacity volumes to host total params."""

    total_params: float
    usable_per_volume: float
    volumes_needed: int
    racks_if_42u: float  # crude: volumes / 42 if one volume per slot

    @staticmethod
    def from_params(
        total_params: float,
        capacity: CapacityParams | None = None,
        slots_per_rack: int = 42,
    ) -> "StoragePlan":
        use = usable_capacity(capacity or CapacityParams())
        if use <= 0:
            raise ValueError("usable capacity must be positive")
        vols = int(math.ceil(total_params / use))
        return StoragePlan(
            total_params=float(total_params),
            usable_per_volume=float(use),
            volumes_needed=vols,
            racks_if_42u=float(vols / slots_per_rack),
        )


@dataclass(frozen=True)
class ThroughputAssumptions:
    """Replace with measured numbers when available.

    digital_active_tflops: effective training TFLOP/s on the digital path
        for the *active* parameter subgraph (experts + non-score ops).
    optical_score_fraction: fraction of dense attention FLOPs attributed to
        QK scores (order-of-magnitude; architecture dependent).
    optical_speedup_on_scores: how many times faster scores run vs digital
        when optical path is live (1.0 = no advantage yet).
    """

    digital_active_tflops: float = 100.0
    optical_score_fraction: float = 0.3
    optical_speedup_on_scores: float = 1.0
    utilization: float = 0.4


@dataclass(frozen=True)
class ComputeEstimate:
    """Coarse training cost from token budget and active size.

    Uses 6 * N_active * tokens FLOPs as a dense-training proxy applied to
    the *active* path (common engineering approximation for MoE).
    """

    flops: float
    effective_tflops: float
    seconds: float
    days: float

    @staticmethod
    def from_budget(
        active_params: float,
        tokens: float,
        thr: ThroughputAssumptions | None = None,
    ) -> "ComputeEstimate":
        thr = thr or ThroughputAssumptions()
        flops = 6.0 * active_params * tokens
        # effective throughput: digital baseline with optional score speedup
        score = thr.optical_score_fraction
        rest = 1.0 - score
        speed = rest + score / max(thr.optical_speedup_on_scores, 1e-9)
        # wall throughput rises when speed factor < 1 (scores faster)
        eff = thr.digital_active_tflops * thr.utilization / max(speed, 1e-9)
        seconds = flops / (eff * 1e12)
        return ComputeEstimate(
            flops=float(flops),
            effective_tflops=float(eff),
            seconds=float(seconds),
            days=float(seconds / 86400.0),
        )


@dataclass(frozen=True)
class ScalePlan:
    """Full plan for one total-parameter target."""

    name: str
    moe: MoEShape
    tokens: TokenBudget
    storage: StoragePlan
    compute: ComputeEstimate
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "moe": asdict(self.moe),
            "tokens": asdict(self.tokens),
            "storage": asdict(self.storage),
            "compute": asdict(self.compute),
            "notes": list(self.notes),
        }


def plan_scale(
    total_params: float,
    *,
    name: str | None = None,
    n_experts: int = 64,
    top_k: int = 2,
    expert_param_frac: float = 0.85,
    token_multiplier: float = 20.0,
    capacity: CapacityParams | None = None,
    thr: ThroughputAssumptions | None = None,
) -> ScalePlan:
    """Build a single scale plan (2e12, 2e13, 2e14, ...)."""
    moe = MoEShape.from_total(
        total_params,
        n_experts=n_experts,
        top_k=top_k,
        expert_param_frac=expert_param_frac,
    )
    tok = TokenBudget.from_active(moe.active_params, multiplier=token_multiplier)
    store = StoragePlan.from_params(total_params, capacity=capacity)
    comp = ComputeEstimate.from_budget(moe.active_params, tok.tokens, thr=thr)
    label = name or _default_name(total_params)
    notes = (
        "Active path drives token budget and FLOP proxy; total params drive storage banks.",
        "Optical score speedup only affects the score fraction of the FLOP model.",
        "Replace ThroughputAssumptions with measured TFLOP/s when hardware exists.",
        "Usable capacity uses capacity.usable_capacity (M#-limited), not geometric ceiling.",
    )
    return ScalePlan(
        name=label,
        moe=moe,
        tokens=tok,
        storage=store,
        compute=comp,
        notes=notes,
    )


def plan_ladder(
    targets: list[float] | None = None,
    **kwargs: Any,
) -> list[ScalePlan]:
    """Same planner at 2T, 20T, 200T by default — comparable rows."""
    if targets is None:
        targets = [2e12, 2e13, 2e14]
    return [plan_scale(t, **kwargs) for t in targets]


def _default_name(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.0f}T"
    if n >= 1e9:
        return f"{n/1e9:.0f}B"
    return f"{n:.0f}"


def format_plan(p: ScalePlan) -> str:
    m, t, s, c = p.moe, p.tokens, p.storage, p.compute
    lines = [
        f"=== {p.name} ===",
        f"total_params     {m.total_params:.4e}",
        f"n_experts/top_k  {m.n_experts}/{m.top_k}",
        f"active_params    {m.active_params:.4e}  ({100*m.active_frac:.2f}% of total)",
        f"token_budget     {t.tokens:.4e}  ({t.multiplier:.1f} x active)",
        f"usable/volume    {s.usable_per_volume:.4e}",
        f"volumes_needed   {s.volumes_needed}",
        f"racks_~42slot    {s.racks_if_42u:.2f}",
        f"train_FLOPs~     {c.flops:.4e}",
        f"eff_TFLOP/s      {c.effective_tflops:.2f}",
        f"wall_days~       {c.days:.2f}  (assumption-limited)",
    ]
    return "\n".join(lines)


def compare_ladder(plans: list[ScalePlan]) -> str:
    """Show that 2T and 200T differ by scale factors, same structure."""
    hdr = f"{'name':8s} {'total':>10s} {'active':>10s} {'tokens':>10s} {'volumes':>8s} {'days~':>10s}"
    rows = [hdr, "-" * len(hdr)]
    for p in plans:
        rows.append(
            f"{p.name:8s} {p.moe.total_params:10.2e} {p.moe.active_params:10.2e} "
            f"{p.tokens.tokens:10.2e} {p.storage.volumes_needed:8d} {p.compute.days:10.1f}"
        )
    if len(plans) >= 2:
        a, b = plans[0], plans[-1]
        rows.append("")
        rows.append(
            f"ratio {b.name}/{a.name}: total={b.moe.total_params/a.moe.total_params:.0f}x  "
            f"active={b.moe.active_params/a.moe.active_params:.0f}x  "
            f"tokens={b.tokens.tokens/a.tokens.tokens:.0f}x  "
            f"volumes={b.storage.volumes_needed/max(a.storage.volumes_needed,1):.0f}x"
        )
    return "\n".join(rows)
