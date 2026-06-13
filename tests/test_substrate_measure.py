"""R28 advisor: unit tests for the shared substrate-measurement
helpers in :mod:`seer.timing.substrate_measure`.

These tests pin the Clopper-Pearson upper, percentile, A2-aligned
threshold count, and the ``safe_write_stub`` overwrite-protection
contract. The CUDA-bound ``measure_block_transfer`` is exercised only
via a smoke test that confirms it raises a deterministic
``RuntimeError`` when CUDA is unavailable -- the real measurement is
covered by the substrate_*.py harness scripts under
``make reproduce-measurement``.
"""
from __future__ import annotations

import json
import math

import pytest

from seer.timing.substrate_measure import (
    MEASURED_STATUS,
    a2_aligned_stats,
    cp_upper_975,
    measure_block_transfer,
    percentile,
    safe_write_stub,
)


# ---------------------------------------------------------------------------
#  cp_upper_975
# ---------------------------------------------------------------------------


def test_cp_upper_k0_n5000_matches_rule_of_three():
    """k=0/n=5000 should give CP upper ~= 3.69/5000 (R20 baseline)."""
    ub = cp_upper_975(0, 5000)
    # The exact scipy beta.ppf(0.975, 1, 5000) ~ 7.378e-4 (slightly
    # tighter than the rule-of-three 6e-4 because we use the
    # two-sided 95% upper, not 95% one-sided 3/n).
    assert 0.0006 < ub < 0.0009


def test_cp_upper_k1_n5000_matches_paper():
    """k=1/n=5000 reproduces the paper's q_step^A2,UB for the 64KB block."""
    ub = cp_upper_975(1, 5000)
    # Paper reports 1.11e-3 at 64KB; the function should land in
    # [1.0e-3, 1.3e-3].
    assert 1.0e-3 < ub < 1.3e-3


def test_cp_upper_k_equals_n_returns_one():
    assert cp_upper_975(5, 5) == 1.0


def test_cp_upper_invalid_n_returns_one():
    assert cp_upper_975(0, 0) == 1.0


def test_cp_upper_monotone_in_k():
    """Increasing k must monotonically increase the upper."""
    ks = [0, 1, 5, 10, 50]
    ubs = [cp_upper_975(k, 5000) for k in ks]
    assert ubs == sorted(ubs)


# ---------------------------------------------------------------------------
#  percentile
# ---------------------------------------------------------------------------


def test_percentile_linear_interpolation():
    xs = list(range(101))  # 0..100 inclusive
    assert percentile(xs, 50) == pytest.approx(50.0)
    assert percentile(xs, 99) == pytest.approx(99.0)
    assert percentile(xs, 99.9) == pytest.approx(99.9, abs=0.01)
    assert percentile(xs, 0) == 0.0
    assert percentile(xs, 100) == 100.0


def test_percentile_empty_returns_nan():
    assert math.isnan(percentile([], 50))


# ---------------------------------------------------------------------------
#  a2_aligned_stats
# ---------------------------------------------------------------------------


def test_a2_aligned_stats_relative_threshold_finds_all_outliers():
    # 5 outliers in 1000 samples at >4*mean.
    base = [10.0] * 1000
    base[:5] = [100.0] * 5  # 4*mean = 4 * (10.45) = 41.8; all 5 over
    s = a2_aligned_stats(base)
    assert s["n_reps"] == 1000
    assert s["n_over_threshold"] == 5
    assert s["q_step_a2_emp"] == pytest.approx(0.005)
    assert s["threshold_kind"] == "relative-4x"
    # Mean dominated by 5 spikes; 4 * 10.45 = 41.8.
    assert 40 < s["threshold_us"] < 45


def test_a2_aligned_stats_fixed_threshold_override():
    base = [10.0] * 1000
    base[:3] = [50.0, 60.0, 70.0]
    s = a2_aligned_stats(base, fixed_threshold_us=45.0)
    assert s["n_over_threshold"] == 3
    assert s["threshold_kind"] == "fixed"
    assert s["threshold_us"] == 45.0


def test_a2_aligned_stats_empty_input():
    s = a2_aligned_stats([])
    assert s["n_reps"] == 0
    assert math.isnan(s["mean_us"])


# ---------------------------------------------------------------------------
#  measure_block_transfer (CUDA-bound; smoke only)
# ---------------------------------------------------------------------------


def test_measure_block_transfer_raises_without_cuda():
    """Without CUDA, the function should raise so callers can fall
    back to ``safe_write_stub``. We check the error message structure
    matches the contract used by substrate_*.py."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    if has_cuda:
        pytest.skip("CUDA is available; cannot test fallback path")
    with pytest.raises(RuntimeError, match=r"torch|CUDA|cuda"):
        measure_block_transfer(block_size_kb=4, n_reps=10, warmup=2)


# ---------------------------------------------------------------------------
#  safe_write_stub overwrite protection
# ---------------------------------------------------------------------------


def test_safe_write_stub_creates_file_when_missing(tmp_path):
    json_p = tmp_path / "x.json"
    tex_p = tmp_path / "x.tex"
    stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN"}
    wrote = safe_write_stub(json_p, tex_p, stub, "TEX CONTENT")
    assert wrote is True
    assert json_p.exists()
    assert json.loads(json_p.read_text())["status"] == "HARNESS_READY_AWAITING_GPU_RUN"
    assert tex_p.read_text() == "TEX CONTENT"


def test_safe_write_stub_refuses_to_clobber_measurement(tmp_path):
    json_p = tmp_path / "x.json"
    tex_p = tmp_path / "x.tex"
    measured = {"status": MEASURED_STATUS, "rows": [{"mean": 11.4}]}
    json_p.write_text(json.dumps(measured))
    tex_p.write_text("MEASURED TEX")
    stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN"}
    wrote = safe_write_stub(json_p, tex_p, stub, "STUB TEX")
    assert wrote is False
    # Original measurement must be untouched.
    assert json.loads(json_p.read_text())["status"] == MEASURED_STATUS
    assert tex_p.read_text() == "MEASURED TEX"


def test_safe_write_stub_force_overrides_measurement(tmp_path):
    json_p = tmp_path / "x.json"
    tex_p = tmp_path / "x.tex"
    measured = {"status": MEASURED_STATUS, "rows": [{"mean": 11.4}]}
    json_p.write_text(json.dumps(measured))
    stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN"}
    wrote = safe_write_stub(json_p, tex_p, stub, "STUB TEX", force=True)
    assert wrote is True
    assert json.loads(json_p.read_text())["status"] == "HARNESS_READY_AWAITING_GPU_RUN"


def test_safe_write_stub_does_not_require_tex(tmp_path):
    json_p = tmp_path / "x.json"
    stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN"}
    wrote = safe_write_stub(json_p, None, stub, None)
    assert wrote is True
    assert json_p.exists()


def test_safe_write_stub_handles_corrupt_existing_json(tmp_path):
    """A non-parseable JSON should not block the stub; we only refuse
    to clobber when we can clearly read MEASURED_STATUS from the
    existing file."""
    json_p = tmp_path / "x.json"
    json_p.write_text("not json at all")
    stub = {"status": "HARNESS_READY_AWAITING_GPU_RUN"}
    wrote = safe_write_stub(json_p, None, stub, None)
    assert wrote is True
    assert json.loads(json_p.read_text())["status"] == "HARNESS_READY_AWAITING_GPU_RUN"
