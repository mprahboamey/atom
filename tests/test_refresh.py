"""Tests for photorefractive readout erase budget stub."""

from atom.refresh import MaterialEraseParams, CrystalBank, default_fe_linbo3_rack


def test_reads_to_eta_min_positive():
    m = MaterialEraseParams()
    n = m.reads_to_eta_min()
    assert n > 0
    assert m.eta_after_reads(n) <= m.eta_min * 1.01


def test_bank_needs_refresh_after_many_reads():
    b = CrystalBank(bank_id=0)
    budget = b.material.reads_to_eta_min()
    b.record_reads(budget)
    assert b.needs_refresh(safety_margin=0.2)


def test_mark_rewritten_resets_clock():
    b = CrystalBank(bank_id=1)
    b.record_reads(100)
    b.mark_rewritten(1.0)
    assert b.read_count == 0.0


def test_rack_plan_refresh():
    rack = default_fe_linbo3_rack(4)
    rack.record_read_on_bank(2, n=10_000)
    jobs = rack.plan_refresh()
    assert any(j["bank_id"] == 2 for j in jobs)
    assert jobs[0]["action"] == "optical_rewrite_in_place"
