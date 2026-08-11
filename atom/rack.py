"""Multi-crystal / multi-rack composition (software planning stub).

Data-center analogy: one GPU holds a slice of a model; many GPUs + a
fabric hold the full model. Here one crystal holds a slice of holographic
weight capacity; N crystals + interconnects compose the model.

Does not simulate photons between chassis. Plans sharding and logical links.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

from .capacity import CapacityParams, capacity_summary, usable_capacity


class ShardMode(str, Enum):
    LAYER_PIPELINE = "layer_pipeline"
    TENSOR_PARALLEL = "tensor_parallel"
    DATA_PARALLEL = "data_parallel"


class InterconnectKind(str, Enum):
    ACTIVATION_BUS = "activation_bus"
    REDUCE_FABRIC = "reduce_fabric"
    CONTROL_AND_WEIGHT_HOST = "control_and_weight_host"
    RACK_BACKPLANE = "rack_backplane"


@dataclass(frozen=True)
class CrystalCapacity:
    params_per_crystal: float
    capacity_report: dict

    @classmethod
    def from_material(cls, params: CapacityParams | None = None) -> "CrystalCapacity":
        p = params or CapacityParams()
        report = capacity_summary(p)
        return cls(params_per_crystal=float(usable_capacity(p)), capacity_report=report)


@dataclass
class ModelBudget:
    n_params: float
    n_layers: int
    hidden_size: int = 4096
    activation_bytes: int = 2

    def activations_per_token(self) -> int:
        return self.hidden_size


@dataclass
class CrystalShard:
    crystal_id: int
    rack_id: int
    mode: ShardMode
    layer_start: int = 0
    layer_end: int = 0
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
    bytes_per_token: float
    note: str = ""


@dataclass
class Rack:
    rack_id: int
    crystals: list[CrystalShard] = field(default_factory=list)


@dataclass
class ClusterPlan:
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
    cap = crystal or CrystalCapacity.from_material()
    if cap.params_per_crystal <= 0:
        raise ValueError("params_per_crystal must be positive")
    return max(1, int(math.ceil(n_params / cap.params_per_crystal)))


def plan_layer_pipeline(
    model: ModelBudget,
    crystal: CrystalCapacity | None = None,
    crystals_per_rack: int = 8,
) -> ClusterPlan:
    crystal = crystal or CrystalCapacity.from_material()
    params_per_layer = model.n_params / max(model.n_layers, 1)
    layers_per_crystal = max(1, int(crystal.params_per_crystal // max(params_per_layer, 1.0)))
    n_crystals = int(math.ceil(model.n_layers / layers_per_crystal))
    n_crystals = max(n_crystals, crystals_needed_for_params(model.n_params, crystal))

    shards: list[CrystalShard] = []
    links: list[InterconnectLink] = []
    layer = 0
    for cid in range(n_crystals):
        start = layer
        end = min(model.n_layers, layer + layers_per_crystal)
        if start >= model.n_layers:
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
            links.append(
                InterconnectLink(
                    kind=InterconnectKind.ACTIVATION_BUS,
                    src_crystal=cid - 1,
                    dst_crystal=cid,
                    bytes_per_token=float(
                        model.activations_per_token() * model.activation_bytes
                    ),
                    note="Residual between pipeline stages",
                )
            )
        layer = end

    for s in shards:
        links.append(
            InterconnectLink(
                kind=InterconnectKind.CONTROL_AND_WEIGHT_HOST,
                src_crystal=-1,
                dst_crystal=s.crystal_id,
                bytes_per_token=0.0,
                note="Host: weight inject, refresh, telemetry",
            )
        )

    racks_map: dict[int, Rack] = {}
    for s in shards:
        racks_map.setdefault(s.rack_id, Rack(rack_id=s.rack_id))
        racks_map[s.rack_id].crystals.append(s)
    racks = [racks_map[k] for k in sorted(racks_map)]

    if len(racks) > 1:
        for a, b in zip(racks, racks[1:]):
            links.append(
                InterconnectLink(
                    kind=InterconnectKind.RACK_BACKPLANE,
                    src_crystal=a.crystals[-1].crystal_id,
                    dst_crystal=b.crystals[0].crystal_id,
                    bytes_per_token=float(
                        model.activations_per_token() * model.activation_bytes
                    ),
                    note="Pipeline crosses rack boundary",
                )
            )

    fits = all(
        s.param_assigned <= s.param_budget + 1e-6
        for s in shards
        if s.layer_end > s.layer_start
    )
    notes = [
        "Layer pipeline: contiguous layer blocks per crystal.",
        "Steady-state interconnect carries activations, not full weights.",
        "Refresh/write path is host to crystal (atom.refresh).",
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
    crystal = crystal or CrystalCapacity.from_material()
    need = crystals_needed_for_params(model.n_params, crystal)
    n = max(n_crystals or need, need)
    params_each = model.n_params / n

    shards = []
    links: list[InterconnectLink] = []
    for cid in range(n):
        shards.append(
            CrystalShard(
                crystal_id=cid,
                rack_id=cid // crystals_per_rack,
                mode=ShardMode.TENSOR_PARALLEL,
                layer_start=0,
                layer_end=model.n_layers,
                tp_rank=cid,
                tp_world=n,
                param_budget=crystal.params_per_crystal,
                param_assigned=params_each,
            )
        )
    for cid in range(n):
        links.append(
            InterconnectLink(
                kind=InterconnectKind.REDUCE_FABRIC,
                src_crystal=cid,
                dst_crystal=(cid + 1) % n,
                bytes_per_token=float(
                    model.activations_per_token() * model.activation_bytes
                ),
                note="TP all-reduce style ring",
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
        "Tensor parallel: weight partitions; all layers on each crystal.",
        "Interconnect is reduce-heavy.",
        f"Assigned ~{params_each:.3e} params/crystal; capacity {crystal.params_per_crystal:.3e}.",
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
    model = ModelBudget(n_params=n_params, n_layers=n_layers, hidden_size=hidden_size)
    crystal = CrystalCapacity.from_material(capacity_params)
    if mode == ShardMode.LAYER_PIPELINE:
        return plan_layer_pipeline(model, crystal, crystals_per_rack=crystals_per_rack)
    if mode in (ShardMode.TENSOR_PARALLEL, ShardMode.DATA_PARALLEL):
        return plan_tensor_parallel(
            model, crystal=crystal, crystals_per_rack=crystals_per_rack
        )
    raise ValueError(f"unknown mode {mode}")
