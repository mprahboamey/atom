"""Multi-crystal / multi-rack composition (software planning stub).

Data-center analogy: one GPU holds a slice of a model; many GPUs + a
fabric hold the full model. Here one crystal (or bank) holds a slice of
holographic weight capacity; N crystals + interconnects compose the model.

This module does not simulate photons between chassis. It answers:
  - How many parameters (or layers) fit in one crystal under capacity.py
  - How to shard a target model across N crystals
  - What must cross the interconnect (activations, not full weights)
  - How racks group crystals and share a digital control plane

Physical write, optical backplane, and measured M# remain open hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from .capacity import CapacityParams, estimate_capacity


class ShardMode(str, Enum):
    """How a transformer is split across crystals."""

    # Crystal i stores layers [i*L:(i+1)*L); activations flow layer-to-layer.
    LAYER_PIPELINE = "layer_pipeline"
    # Crystal i stores a partition of each layer's weights (e.g. head groups).
    TENSOR_PARALLEL = "tensor_parallel"
    # Independent replicas (same weights); used for batch parallel, not model scale.
    DATA_PARALLEL = "data_parallel"


class InterconnectKind(str, Enum):
    """Logical link type between crystals / racks.

    Weights stay in-crystal during steady inference (except refresh writes).
    What moves on the wire is much smaller: activations, norms, control.
    """

    # Residual stream / hidden state between pipeline stages (layer shards).
    ACTIVATION_BUS = "activation_bus"
    # Partial attention or MLP results to reduce in tensor parallel.
    REDUCE_FABRIC = "reduce_fabric"
    # Host <-> crystal: load weights, refresh rewrite, telemetry.
    CONTROL_AND_WEIGHT_HOST = "control_and_weight_host"
    # Rack-to-rack aggregation (same roles, longer reach).
    RACK_BACKPLANE = "rack_backplane"


@dataclass(frozen=True)
class CrystalCapacity:
    """Usable weight slots for one crystal volume."""

    params_per_crystal: float
    capacity_report: dict

    @classmethod
    def from_material(cls, params: CapacityParams | None = None) -> "CrystalCapacity":
        p = params or CapacityParams()
        report = estimate_capacity(p)
        return cls(params_per_crystal=float(report["usable_params"]), capacity_report=report)


@dataclass
class ModelBudget:
    """Target model size for placement."""

    n_params: float
    n_layers: int
    hidden_size: int = 4096
    # Bytes per activation element when estimating interconnect traffic (fp16).
    activation_bytes: int = 2

    def activations_per_token(self) -> int:
        """Hidden residual size in elements (one position)."""
        return self.hidden_size


@dataclass
class CrystalShard:
    crystal_id: int
    rack_id: int
    mode: ShardMode
    # Inclusive layer range for pipeline mode; empty if unused.
    layer_start: int = 0
    layer_end: int = 0
    # Tensor-parallel rank and world size.
    tp_rank: int = 0
    tp_world: int = 1
    param_budget: float = 0.0
    param_assigned: float = 0.0

    @property
    def layers(self) -> range:
        return range(self.layer_start, self.layer_end)


@dataclass
class InterconnectLink:
    kind: InterconnectKind
    src_crystal: int
    dst_crystal: int
    # Rough traffic per token forwarded on this link (bytes).
    bytes_per_token: float
    note: str = ""


@dataclass
class Rack:
    rack_id: int
    crystals: list[CrystalShard] = field(default_factory=list)


@dataclass
class ClusterPlan:
    """Full placement: model spread over crystals and racks."""

    model: ModelBudget
    mode: ShardMode
    crystal_capacity: CrystalCapacity
    racks: list[Rack]
    links: list[InterconnectLink]
    n_crystals_required: int
    fits: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "mode": self.mode.value,
            "n_params": self.model.n_params,
            "n_layers": self.model.n_layers,
            "params_per_crystal": self.crystal_capacity.params_per_crystal,
            "n_crystals_required": self.n_crystals_required,
            "n_racks": len(self.racks),
            "fits": self.fits,
            "n_links": len(self.links),
            "link_kinds": sorted({lnk.kind.value for lnk in self.links}),
            "notes": self.notes,
        }


def crystals_needed_for_params(
    n_params: float,
    crystal: CrystalCapacity | None = None,
) -> int:
    """Minimum crystals so sum of usable params >= n_params."""
    cap = crystal or CrystalCapacity.from_material()
    if cap.params_per_crystal <= 0:
        raise ValueError("params_per_crystal must be positive")
    return max(1, int(math.ceil(n_params / cap.params_per_crystal)))


def plan_layer_pipeline(
    model: ModelBudget,
    crystal: CrystalCapacity | None = None,
    crystals_per_rack: int = 8,
) -> ClusterPlan:
    """Shard by contiguous layer blocks (pipeline parallel).

    Interconnect: activation residual between stage i and i+1 only.
    """
    crystal = crystal or CrystalCapacity.from_material()
    # Capacity can also limit layers if each layer has ~n_params/n_layers weights.
    params_per_layer = model.n_params / max(model.n_layers, 1)
    layers_per_crystal = max(1, int(crystal.params_per_crystal // params_per_layer))
    n_crystals = int(math.ceil(model.n_layers / layers_per_crystal))
    # Also respect pure param ceiling
    n_crystals = max(n_crystals, crystals_needed_for_params(model.n_params, crystal))

    shards: list[CrystalShard] = []
    links: list[InterconnectLink] = []
    layer = 0
    for cid in range(n_crystals):
        start = layer
        end = min(model.n_layers, layer + layers_per_crystal)
        if start >= model.n_layers:
            # Extra crystals only for param overflow: keep empty layer range
            start, end = model.n_layers, model.n_layers
        rack_id = cid // crystals_per_rack
        assigned = (end - start) * params_per_layer
        shards.append(
            CrystalShard(
                crystal_id=cid,
                rack_id=rack_id,
                mode=ShardMode.LAYER_PIPELINE,
                layer_start=start,
                layer_end=end,
                param_budget=crystal.params_per_crystal,
                param_assigned=assigned,
            )
        )
        if cid > 0 and end > start:
            prev = cid - 1
            links.append(
                InterconnectLink(
                    kind=InterconnectKind.ACTIVATION_BUS,
                    src_crystal=prev,
                    dst_crystal=cid,
                    bytes_per_token=float(model.activations_per_token() * model.activation_bytes),
                    note="Residual / hidden state between pipeline stages",
                )
            )
        layer = end

    # Host control plane to every crystal (weight load + refresh)
    for s in shards:
        links.append(
            InterconnectLink(
                kind=InterconnectKind.CONTROL_AND_WEIGHT_HOST,
                src_crystal=-1,
                dst_crystal=s.crystal_id,
                bytes_per_token=0.0,
                note="Digital host: weight inject, refresh, telemetry (not per-token)",
            )
        )

    racks_map: dict[int, Rack] = {}
    for s in shards:
        racks_map.setdefault(s.rack_id, Rack(rack_id=s.rack_id))
        racks_map[s.rack_id].crystals.append(s)
    racks = [racks_map[k] for k in sorted(racks_map)]

    # Rack backplane if more than one rack
    if len(racks) > 1:
        for a, b in zip(racks, racks[1:]):
            links.append(
                InterconnectLink(
                    kind=InterconnectKind.RACK_BACKPLANE,
                    src_crystal=a.crystals[-1].crystal_id,
                    dst_crystal=b.crystals[0].crystal_id,
                    bytes_per_token=float(model.activations_per_token() * model.activation_bytes),
                    note="Pipeline crosses rack boundary",
                )
            )

    fits = all(s.param_assigned <= s.param_budget + 1e-6 for s in shards if s.layer_end > s.layer_start)
    notes = [
        "Layer pipeline: each crystal holds a contiguous layer block.",
        "Steady-state interconnect carries activations, not full weights.",
        "Refresh/write path is host→crystal (see atom.refresh).",
    ]
    return ClusterPlan(
        model=model,
        mode=ShardMode.LAYER_PIPELINE,
        crystal_capacity=crystal,
        racks=racks,
        links=links,
        n_crystals_required=n_crystals,
        fits=fits,
        notes=notes,
    )


def plan_tensor_parallel(
    model: ModelBudget,
    n_crystals: int | None = None,
    crystal: CrystalCapacity | None = None,
    crystals_per_rack: int = 8,
) -> ClusterPlan:
    """Shard weight partitions inside each layer across crystals (TP).

    Every crystal sees every layer's activation slice; partial results
    reduce over the fabric each layer.
    """
    crystal = crystal or CrystalCapacity.from_material()
    need = crystals_needed_for_params(model.n_params, crystal)
    n = max(n_crystals or need, need)
    params_each = model.n_params / n

    shards = []
    links: list[InterconnectLink] = []
    for cid in range(n):
        rack_id = cid // crystals_per_rack
        shards.append(
            CrystalShard(
                crystal_id=cid,
                rack_id=rack_id,
                mode=ShardMode.TENSOR_PARALLEL,
                layer_start=0,
                layer_end=model.n_layers,
                tp_rank=cid,
                tp_world=n,
                param_budget=crystal.params_per_crystal,
                param_assigned=params_each,
            )
        )
    # All-to-all style reduce fabric (logical mesh edges as ring for stub)
    for cid in range(n):
        nxt = (cid + 1) % n
        links.append(
            InterconnectLink(
                kind=InterconnectKind.REDUCE_FABRIC,
                src_crystal=cid,
                dst_crystal=nxt,
                bytes_per_token=float(model.activations_per_token() * model.activation_bytes),
                note="TP partial-sum / all-reduce of activations or attn fragments",
            )
        )
        links.append(
            InterconnectLink(
                kind=InterconnectKind.CONTROL_AND_WEIGHT_HOST,
                src_crystal=-1,
                dst_crystal=cid,
                bytes_per_token=0.0,
                note="Host weight inject and refresh",
            )
        )

    racks_map: dict[int, Rack] = {}
    for s in shards:
        racks_map.setdefault(s.rack_id, Rack(rack_id=s.rack_id))
        racks_map[s.rack_id].crystals.append(s)
    racks = [racks_map[k] for k in sorted(racks_map)]

    fits = params_each <= crystal.params_per_crystal + 1e-6
    notes = [
        "Tensor parallel: weights partitioned; all layers present on each crystal.",
        "Interconnect is reduce-heavy (like GPU NVLink all-reduce).",
        f"Assigned ~{params_each:.3e} params per crystal; capacity {crystal.params_per_crystal:.3e}.",
    ]
    return ClusterPlan(
        model=model,
        mode=ShardMode.TENSOR_PARALLEL,
        crystal_capacity=crystal,
        racks=racks,
        links=links,
        n_crystals_required=n,
        fits=fits,
        notes=notes,
    )


def plan_cluster(
    n_params: float,
    n_layers: int,
    mode: ShardMode = ShardMode.LAYER_PIPELINE,
    hidden_size: int = 4096,
    crystals_per_rack: int = 8,
    capacity_params: CapacityParams | None = None,
) -> ClusterPlan:
    """Entry point: place a model on a crystal cluster."""
    model = ModelBudget(n_params=n_params, n_layers=n_layers, hidden_size=hidden_size)
    crystal = CrystalCapacity.from_material(capacity_params)
    if mode == ShardMode.LAYER_PIPELINE:
        return plan_layer_pipeline(model, crystal, crystals_per_rack=crystals_per_rack)
    if mode == ShardMode.TENSOR_PARALLEL:
        return plan_tensor_parallel(model, crystal=crystal, crystals_per_rack=crystals_per_rack)
    if mode == ShardMode.DATA_PARALLEL:
        # One full copy per crystal; N is chosen by caller via capacity only
        n = crystals_needed_for_params(n_params, crystal)
        return plan_tensor_parallel(model, n_crystals=n, crystal=crystal, crystals_per_rack=crystals_per_rack)
    raise ValueError(f"unknown mode {mode}")
