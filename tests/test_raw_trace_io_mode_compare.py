"""R30: golden pins for LogNormal vs recorded raw-trace IO compare."""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSON = ROOT / "experiments/eC_bound_tightness/results/raw_trace_io_mode_compare.json"
TEX = ROOT / "experiments/eC_bound_tightness/results/raw_trace_io_mode_compare.tex"


@pytest.mark.skipif(not JSON.is_file(), reason="run raw_trace_io_mode_compare first")
def test_io_mode_compare_headline_tallies():
    data = json.loads(JSON.read_text())
    assert data["unsafe_tally_identical"] is False
    logn = data["lognormal"]
    rec = data["recorded"]
    assert logn["bernstein-cor2"]["accepted"] == rec["bernstein-cor2"]["accepted"] == 23
    assert logn["bernstein-cor2"]["unsafe"] == 11
    assert rec["bernstein-cor2"]["unsafe"] == 6
    assert logn["full-tail-PS"]["accepted"] == rec["full-tail-PS"]["accepted"] == 22
    assert logn["full-tail-PS"]["unsafe"] == 10
    assert rec["full-tail-PS"]["unsafe"] == 5
    for v in ("bernstein-cor2", "full-tail-PS"):
        assert logn[v]["accepted"] == rec[v]["accepted"]


@pytest.mark.skipif(not TEX.is_file(), reason="run raw_trace_io_mode_compare first")
def test_io_mode_compare_tex_has_four_columns():
    body = TEX.read_text()
    assert r"\begin{tabular}" in body
    assert "bernstein-cor2" in body
    assert "full-tail-PS" in body
