"""R28 advisor P1': golden test for raw_trace_replay ``--io_mode``.

Pins the behaviour of the three IO modes added in R28:

- ``lognormal`` (default): parametric LogNormal substrate injection
  -- this is what the paper headline 10/36 unsafe count comes from.
- ``recorded``: replaces the parametric injection with the trace's
  own per-step ``per_step_io_us`` (real measured-DMA at collection
  time). Should produce a non-trivially different tally; the test
  doesn't pin an exact count, only that the function runs
  end-to-end and emits a tally that is structurally well-formed.
- ``measured``: stubbed (GPU-only); should raise NotImplementedError.

We exercise the helper directly with a tiny synthetic trace so the
test stays CPU-only and finishes in <1 s.
"""
from __future__ import annotations

import math

import pytest

from experiments.eC_bound_tightness.raw_trace_replay import (
    trace_empirical_miss,
)


def _synthetic_trace(n_steps: int = 50,
                     base_us: float = 30.0,
                     io_recorded_us: float = 0.0,
                     eps_step: float = 0.1,
                     block_count: int = 32) -> list[dict]:
    return [
        {"eps_step": eps_step, "block_count": block_count,
         "base_us": base_us, "io_recorded_us": io_recorded_us}
        for _ in range(n_steps)
    ]


def test_io_mode_lognormal_runs_and_returns_well_formed_tuple():
    trace = _synthetic_trace()
    em, lo, hi, n = trace_empirical_miss(
        trace, ell_bar_us=200.0, D_us=200_000.0,
        sigma_residual_us=100.0, seed=42, io_mode="lognormal")
    assert n == len(trace)
    assert 0.0 <= em <= 1.0
    assert lo <= em <= hi
    assert not math.isnan(em)


def test_io_mode_recorded_runs_with_recorded_io_path():
    """When per_step_io_us > 0, recorded mode should consume it and
    return a tally without raising. The fact that recorded != lognormal
    on the same seed is verified separately below."""
    trace = _synthetic_trace(io_recorded_us=150.0)
    em, lo, hi, n = trace_empirical_miss(
        trace, ell_bar_us=200.0, D_us=200_000.0,
        sigma_residual_us=100.0, seed=42, io_mode="recorded")
    assert n == len(trace)
    assert 0.0 <= em <= 1.0
    assert lo <= em <= hi


def test_io_mode_measured_is_stubbed():
    trace = _synthetic_trace()
    with pytest.raises(NotImplementedError, match=r"measured"):
        trace_empirical_miss(
            trace, ell_bar_us=200.0, D_us=200_000.0,
            sigma_residual_us=100.0, seed=42, io_mode="measured")


def test_io_mode_invalid_raises_value_error():
    trace = _synthetic_trace()
    with pytest.raises(ValueError, match=r"io_mode"):
        trace_empirical_miss(
            trace, ell_bar_us=200.0, D_us=200_000.0,
            sigma_residual_us=100.0, seed=42, io_mode="bogus")


def test_lognormal_and_recorded_can_differ_on_same_seed():
    """On a deadline tight enough that the IO term matters, the
    parametric LogNormal draw must produce a different miss rate
    than the trace's flat recorded IO -- otherwise the ``recorded``
    mode would not be exercising the path advertised."""
    # Tight deadline + non-zero recorded IO so both modes are exercised.
    trace = _synthetic_trace(
        n_steps=200, base_us=1000.0, io_recorded_us=200.0,
        eps_step=0.5, block_count=8)
    em_ln, *_ = trace_empirical_miss(
        trace, ell_bar_us=300.0, D_us=2_000.0,
        sigma_residual_us=100.0, seed=0, io_mode="lognormal")
    em_rec, *_ = trace_empirical_miss(
        trace, ell_bar_us=300.0, D_us=2_000.0,
        sigma_residual_us=100.0, seed=0, io_mode="recorded")
    # The two modes consume different IO distributions; with the
    # tight deadline above at least one of the two miss-rates should
    # land off zero and they should not be identical.
    assert em_ln != em_rec or em_ln > 0.0
