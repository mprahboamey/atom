"""Optical vs digital greedy identity on a real safetensors model.

Skipped unless ATOM_SAFETENSORS_MODEL points at an HF folder
(config.json + model.safetensors).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL = os.environ.get("ATOM_SAFETENSORS_MODEL", "")


@pytest.mark.skipif(
    not MODEL or not Path(MODEL).exists(),
    reason="Set ATOM_SAFETENSORS_MODEL to an HF safetensors folder to run",
)
def test_optical_digital_generate_identical():
    from atom.hybrid_model import HybridTransformer

    model = HybridTransformer.from_checkpoint(MODEL, stream_layers=True)
    prompt = [1, 2, 3, 4]
    opt = model.generate(prompt, max_new_tokens=3, use_optical=True)
    dig = model.generate(prompt, max_new_tokens=3, use_optical=False)
    assert opt == dig


def test_hybrid_import():
    from atom.hybrid_model import HybridTransformer, ModelConfig

    assert HybridTransformer is not None
    assert ModelConfig().n_layers == 32
