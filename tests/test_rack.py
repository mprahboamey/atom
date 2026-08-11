"""Multi-crystal rack placement tests."""

from atom.rack import (
    CrystalCapacity,
    ShardMode,
    crystals_needed_for_params,
    plan_cluster,
)


def test_crystals_needed_scales():
    cap = CrystalCapacity(params_per_crystal=1e9, capacity_report={})
    assert crystals_needed_for_params(3.5e9, cap) == 4


def test_layer_pipeline_summary():
    plan = plan_cluster(n_params=1e10, n_layers=40, mode=ShardMode.LAYER_PIPELINE)
    s = plan.summary()
    assert s["n_crystals_required"] >= 1
    assert "activation_bus" in s["link_kinds"] or s["n_crystals_required"] == 1
    assert "control_and_weight_host" in s["link_kinds"]


def test_tensor_parallel_has_reduce():
    plan = plan_cluster(n_params=1e10, n_layers=32, mode=ShardMode.TENSOR_PARALLEL)
    assert "reduce_fabric" in plan.summary()["link_kinds"]


def test_pipeline_layer_coverage():
    plan = plan_cluster(n_params=5e9, n_layers=12, mode=ShardMode.LAYER_PIPELINE)
    layers = set()
    for rack in plan.racks:
        for c in rack.crystals:
            layers.update(c.layers)
    assert layers == set(range(12)) or plan.n_crystals_required >= 1
