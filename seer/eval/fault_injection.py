"""Fault-mode injection harness for SEER (review O5 / R4).

The bound's correctness depends on:
  * The LAP predictor producing useful probabilities (not random noise)
  * The PI controller integrating finite latencies (not NaN / Inf)
  * The online σ-tracker republishing under shift, not under garbage
  * The DMA tier being roughly stationary in p99 latency

Reviewers ask: *what happens when one of these inputs becomes adversarial?*
This module provides 5 injection wrappers + a measurement harness that
returns, for each fault f:

  * ``recovery_steps`` — # decode steps from injection to miss-ratio
    returning within 1.5× of the pre-fault baseline
  * ``miss_spike`` — peak miss-ratio during the fault window
  * ``bound_valid`` — whether the Lemma-2 bound still upper-bounds the
    observed miss-ratio (T11)

The wrappers compose as context managers so the underlying policy /
controller code is not modified.
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
#  Fault 1: LAP TRT plan corrupt — replace predictions with random uniform.
# ---------------------------------------------------------------------------


@contextmanager
def lap_corrupt(predictor: Any, start_step: int, duration_steps: int,
                tracker: dict | None = None):
    """Within [start_step, start_step+duration_steps), replace LAP output
    with U[0,1] noise. The SafeFallback wrapper should detect the
    confidence drop and route to H2O.
    """
    orig_call = predictor.__call__ if hasattr(predictor, "__call__") else None
    if orig_call is None:
        raise RuntimeError("predictor not callable")
    step_box = {"step": 0, "corrupted_calls": 0}

    def wrapped(X: np.ndarray) -> np.ndarray:
        cur = step_box["step"]
        if start_step <= cur < start_step + duration_steps:
            step_box["corrupted_calls"] += 1
            # uniform in [0,1] preserves shape; LAP confidence collapses
            return np.random.RandomState(cur).uniform(0, 1, size=(X.shape[0], 4)).astype(np.float32)
        return orig_call(X)

    type_of = type(predictor)
    predictor.__seer_fault_step__ = step_box
    predictor.__seer_fault_orig__ = orig_call
    # Monkey-patch __call__
    if hasattr(predictor, "_call_impl"):
        # torch module path — patch _call_impl instead.
        predictor._call_impl = lambda *a, **k: wrapped(*a, **k)
    else:
        # plain callable — replace the __call__ attribute via a wrapper
        predictor.__seer_wrapped_call__ = wrapped
        # SEERPolicy invokes `self.lap(X)` which routes to __call__.
        # We rebind by setting an instance attribute that shadows class __call__:
        # Python special-method lookup goes through the class, so we monkey
        # the bound method instead.
        predictor.__class__ = type(type_of.__name__ + "_Faulty", (type_of,), {
            "__call__": lambda self, X: self.__seer_wrapped_call__(X)
        })
    try:
        yield step_box
    finally:
        # restore
        if hasattr(predictor, "_call_impl"):
            try:
                del predictor._call_impl
            except AttributeError:
                pass
        if isinstance(predictor.__class__.__name__, str) and \
           predictor.__class__.__name__.endswith("_Faulty"):
            predictor.__class__ = type_of
        if tracker is not None:
            tracker["lap_corrupt_calls"] = step_box["corrupted_calls"]


# ---------------------------------------------------------------------------
#  Fault 2: CUDA OOM — preallocate a hog tensor that holds memory.
# ---------------------------------------------------------------------------


@contextmanager
def cuda_oom_hog(reserve_gb: float = 70.0):
    """Allocate ``reserve_gb`` of HBM and hold it for the duration of the
    block. Subsequent torch allocations that exceed the residual budget
    should raise ``torch.cuda.OutOfMemoryError``, which production code
    must catch and degrade.
    """
    import torch
    if not torch.cuda.is_available():
        yield None
        return
    hog = None
    try:
        elems = int(reserve_gb * (1024**3) / 2)  # fp16
        try:
            hog = torch.empty(elems, dtype=torch.float16, device="cuda")
        except torch.cuda.OutOfMemoryError:
            hog = None  # the hog itself OOMs — still simulates pressure
        yield hog
    finally:
        if hog is not None:
            del hog
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  Fault 3: PI controller NaN injection — measure that controller is robust.
# ---------------------------------------------------------------------------


def inject_pi_nan(controller: Any, n_nan_samples: int = 50) -> dict:
    """Feed ``n_nan_samples`` NaN observations to a LambdaController.

    A non-robust implementation would propagate NaN through λ, then
    every utility becomes NaN and every selection becomes garbage.
    Our hardened :class:`LambdaController` drops NaN samples and
    increments ``corrupted_samples``. The returned dict reports
    the controller's pre / post state.
    """
    pre = controller.stats()
    pre_lambda = pre["lambda"]
    for _ in range(n_nan_samples):
        controller.observe(float("nan"))
    post = controller.stats()
    post_lambda = post["lambda"]
    return {
        "pre_lambda": pre_lambda,
        "post_lambda": post_lambda,
        "delta": post_lambda - pre_lambda,
        "corrupted_dropped": post["corrupted_samples"] - pre.get("corrupted_samples", 0),
        "lambda_finite": math.isfinite(post_lambda),
    }


# ---------------------------------------------------------------------------
#  Fault 4: σ-tracker drift — feed shifted workload, verify mode switch.
# ---------------------------------------------------------------------------


def test_sigma_drift(tracker: Any, baseline_latency_us: float,
                     shift_factor: float = 5.0, n_per_phase: int = 200) -> dict:
    """Push ``n_per_phase`` clean samples ≈ baseline, then ``n_per_phase``
    samples around ``baseline * shift_factor``. The tracker should
    flip to the truncated heavy-tail bound after the shift (q > q_threshold).
    """
    # Phase A: clean
    rng = random.Random(0)
    for _ in range(n_per_phase):
        tracker.observe(baseline_latency_us + rng.gauss(0, baseline_latency_us * 0.05))
    state_a = tracker.state
    # Phase B: shift
    for _ in range(n_per_phase):
        # mix shifted + tail
        if rng.random() < 0.1:
            tracker.observe(baseline_latency_us * shift_factor * 5)  # extreme tail
        else:
            tracker.observe(baseline_latency_us * shift_factor + rng.gauss(0, baseline_latency_us * 0.3))
    state_b = tracker.state
    return {
        "phase_a": {
            "sigma_us": state_a.sigma_us if state_a else None,
            "q": state_a.q if state_a else None,
            "using_truncated": state_a.using_truncated if state_a else None,
        },
        "phase_b": {
            "sigma_us": state_b.sigma_us if state_b else None,
            "q": state_b.q if state_b else None,
            "using_truncated": state_b.using_truncated if state_b else None,
        },
        "switched_to_truncated": bool(state_b and state_b.using_truncated and
                                      (not state_a or not state_a.using_truncated)),
    }


# ---------------------------------------------------------------------------
#  Fault 5: NVMe contention burst — inject artificial DMA delay.
# ---------------------------------------------------------------------------


@contextmanager
def nvme_burst(simulator: Any, start_step: int, duration_steps: int,
               extra_us_per_block: float = 5000.0):
    """Within the burst window, add ``extra_us_per_block`` to the IO
    penalty. Simulates fio contention / NVMe GC pause.

    Patches :meth:`MaskingSimulator._measure_dma_per_block_us` to add a
    constant delay during the window.
    """
    if not hasattr(simulator, "_measure_dma_per_block_us"):
        yield {"applied": 0}
        return
    orig = simulator._measure_dma_per_block_us
    step_box = {"step": 0, "applied": 0, "extra_us": extra_us_per_block}

    def patched(n_blocks: int) -> float | None:
        cur = step_box["step"]
        base = orig(n_blocks)
        if base is None:
            return None
        if start_step <= cur < start_step + duration_steps:
            step_box["applied"] += 1
            return base + extra_us_per_block
        return base

    simulator._measure_dma_per_block_us = patched  # type: ignore[method-assign]
    try:
        yield step_box
    finally:
        simulator._measure_dma_per_block_us = orig  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
#  Recovery measurement helper.
# ---------------------------------------------------------------------------


@dataclass
class FaultMeasurement:
    """Outcome of one fault-mode trial.

    Fields ----------
    name : human-readable fault label
    pre_miss : miss-ratio over the pre-fault window
    fault_miss : miss-ratio during the fault window
    post_miss : miss-ratio over the post-fault recovery window
    miss_spike : fault_miss / max(pre_miss, 1e-4) ratio
    recovery_steps : steps from fault end to miss-ratio returning to
                     within 1.5x of pre_miss
    bound_valid : whether the Lemma-2 bound still upper-bounds fault_miss
    notes : any side-channel observations (corrupted counts, fallback events)
    """
    name: str
    pre_miss: float
    fault_miss: float
    post_miss: float
    miss_spike: float
    recovery_steps: int
    bound_valid: bool
    notes: dict


def measure_recovery_window(per_step_us: list[float], deadline_us: float,
                            fault_start: int, fault_end: int,
                            pre_window: int = 200, post_window: int = 400,
                            recovery_tol: float = 1.5,
                            bound_value: float | None = None) -> FaultMeasurement:
    """Slice a latency vector around a fault window and compute the
    pre / fault / post miss-ratios + recovery step count.
    """
    def miss_ratio(xs: list[float]) -> float:
        if not xs:
            return 0.0
        return sum(1 for x in xs if x > deadline_us) / len(xs)

    pre = per_step_us[max(0, fault_start - pre_window):fault_start]
    fault = per_step_us[fault_start:fault_end]
    post = per_step_us[fault_end:fault_end + post_window]
    pre_miss = miss_ratio(pre)
    fault_miss = miss_ratio(fault)
    post_miss = miss_ratio(post)
    spike = fault_miss / max(pre_miss, 1e-4)
    # recovery: scan a sliding 50-step window starting at fault_end,
    # find the first window with miss-ratio <= recovery_tol * pre_miss
    target = recovery_tol * max(pre_miss, 1e-4)
    recovery_steps = -1
    for s in range(fault_end, min(len(per_step_us), fault_end + post_window) - 50):
        w = per_step_us[s:s + 50]
        if miss_ratio(w) <= target:
            recovery_steps = s - fault_end
            break
    bound_valid = (bound_value is None) or (fault_miss <= bound_value + 1e-6)
    return FaultMeasurement(
        name="", pre_miss=pre_miss, fault_miss=fault_miss, post_miss=post_miss,
        miss_spike=spike, recovery_steps=recovery_steps,
        bound_valid=bound_valid, notes={},
    )
