"""Shared substrate-measurement helpers (R28 advisor consolidation).

Single source of truth for the four ``experiments/eC_bound_tightness/
substrate_*.py`` scripts + ``seer/timing/a7_probe.py``. Each of those
scripts used to inline its own copy of the Clopper-Pearson upper,
linear-interpolated percentile, A2-aligned threshold counter, and
harness-stub writer; this module collapses them to a single
authoritative copy so per-block statistics cannot drift across the
substrate / contended / moderate / MPS variants. The contender life-
cycle (threading vs. subprocess) is kept inside the variant scripts
because each one differs structurally; only the shared statistics
helpers live here.

Public API:

- ``cp_upper_975(k, n)``: two-sided 95% Clopper-Pearson upper.
- ``percentile(xs, p)``: linear-interpolated percentile helper.
- ``a2_aligned_stats(timings_us, fixed_threshold_us=None)``: returns
  the canonical per-block summary dict (mean / median / P99 / P99.9 /
  max / k=n_over_threshold / q_step_a2_emp / q_step_a2_upper95).
- ``measure_block_transfer(block_size_kb, n_reps, warmup,
  device_index=0)``: bare fp16 H2D+D2H ``cudaMemcpyAsync`` micro-
  benchmark on a dedicated stream. Returns the raw timings list (caller
  applies ``a2_aligned_stats``).
- ``safe_write_stub(json_path, tex_path, stub_dict, tex_content,
  *, force=False)``: writes a harness-only stub iff the artifact has
  not already been populated by a measurement run. Refuses to over-
  write a JSON whose top-level ``status`` is ``MEASUREMENT_COMPLETE``
  unless ``force=True`` is passed; returns ``True`` if the write
  happened, ``False`` if the existing measured artifact was preserved.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics


def cp_upper_975(k: int, n: int) -> float:
    """Two-sided 95% Clopper-Pearson upper at ``(k, n)``.

    Matches the convention used across the paper (``beta.ppf(0.975,
    k+1, n-k)``). Falls back to a chi-square approximation when SciPy
    is unavailable so the module remains importable in CPU-only
    smoke environments.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta as _beta  # type: ignore
        return float(_beta.ppf(0.975, k + 1, max(1, n - k)))
    except Exception:
        chi2_table_975 = {
            2: 7.378, 4: 11.143, 6: 14.449, 8: 17.535,
            10: 20.483, 12: 23.337, 14: 26.119, 16: 28.845,
            18: 31.526, 20: 34.170,
        }
        df = 2 * (k + 1)
        chi2_v = chi2_table_975.get(
            df, df + 2.0 * math.sqrt(2.0 * df) + 2.0)
        return chi2_v / (2.0 * n)


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (NumPy's default 'linear' rule)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (p / 100.0) * (len(s) - 1)
    lo = math.floor(k)
    hi = min(len(s) - 1, lo + 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def a2_aligned_stats(
    timings_us: list[float],
    *,
    fixed_threshold_us: float | None = None,
) -> dict:
    """Canonical per-block summary used by every substrate script.

    ``timings_us`` is the raw per-rep H2D+D2H elapsed-time list (us).
    By default the A2-aligned threshold is the relative
    :math:`4\\bar\\ell` truncation point of Lemma 2 (matches
    ``substrate_pcie_nvme`` and ``a7_probe``). Passing
    ``fixed_threshold_us`` overrides this with an absolute threshold,
    which is what the ``substrate_a7_moderate`` and
    ``substrate_a7_mps_isolated`` scripts use to compare contended
    runs against a quiesced-phase reference.
    """
    if not timings_us:
        return {
            "n_reps": 0, "mean_us": float("nan"), "median_us": float("nan"),
            "p99_us": float("nan"), "p999_us": float("nan"),
            "max_us": float("nan"), "n_over_threshold": 0,
            "threshold_us": float("nan"), "threshold_kind": "relative-4x",
            "q_step_a2_emp": float("nan"),
            "q_step_a2_upper95": float("nan"),
        }
    n = len(timings_us)
    mean = statistics.mean(timings_us)
    if fixed_threshold_us is None:
        threshold = 4.0 * mean
        threshold_kind = "relative-4x"
    else:
        threshold = float(fixed_threshold_us)
        threshold_kind = "fixed"
    n_over = sum(1 for t in timings_us if t > threshold)
    return {
        "n_reps": n,
        "mean_us": mean,
        "median_us": statistics.median(timings_us),
        "p99_us": percentile(timings_us, 99.0),
        "p999_us": percentile(timings_us, 99.9),
        "max_us": max(timings_us),
        "n_over_threshold": n_over,
        "threshold_us": threshold,
        "threshold_kind": threshold_kind,
        "q_step_a2_emp": n_over / n,
        "q_step_a2_upper95": cp_upper_975(n_over, n),
    }


def measure_block_transfer(
    block_size_kb: int,
    n_reps: int,
    warmup: int,
    device_index: int = 0,
) -> list[float]:
    """Bare fp16 H2D+D2H ``cudaMemcpyAsync`` microbenchmark.

    Returns the raw per-rep elapsed-time list in microseconds. Callers
    apply :func:`a2_aligned_stats` to derive summary stats. Requires
    CUDA + PyTorch; raises ``RuntimeError`` otherwise so callers can
    fall back to a harness stub via :func:`safe_write_stub`.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"torch import failed: {exc!r}") from exc
    if not torch.cuda.is_available():  # pragma: no cover - HW guard
        raise RuntimeError("torch.cuda.is_available() returned False")

    dev = torch.device("cuda", device_index)
    n_bytes = block_size_kb * 1024
    nb_elem = n_bytes // 2  # fp16
    host = torch.empty(nb_elem, dtype=torch.float16, pin_memory=True)
    devv = torch.empty(nb_elem, dtype=torch.float16, device=dev)
    stream = torch.cuda.Stream(device=dev)

    for _ in range(warmup):
        with torch.cuda.stream(stream):
            devv.copy_(host, non_blocking=True)
            host.copy_(devv, non_blocking=True)
    torch.cuda.synchronize(device=dev)

    timings: list[float] = []
    for _ in range(n_reps):
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            s_evt.record(stream)
            devv.copy_(host, non_blocking=True)
            host.copy_(devv, non_blocking=True)
            e_evt.record(stream)
        e_evt.synchronize()
        timings.append(s_evt.elapsed_time(e_evt) * 1000.0)
    return timings


# Status sentinel that indicates an artifact carries real measured
# data. ``safe_write_stub`` refuses to overwrite a JSON whose top-
# level ``status`` field matches this value (unless ``force=True``).
MEASURED_STATUS = "MEASUREMENT_COMPLETE"


def _is_measured(json_path: pathlib.Path) -> bool:
    if not json_path.exists():
        return False
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return data.get("status") == MEASURED_STATUS


def safe_write_stub(
    json_path: pathlib.Path,
    tex_path: pathlib.Path | None,
    stub_dict: dict,
    tex_content: str | None,
    *,
    force: bool = False,
) -> bool:
    """Write a harness-only stub iff ``json_path`` is not measured.

    The five substrate harnesses
    (``substrate_pcie_nvme``, ``substrate_a7_contended``,
    ``substrate_a7_moderate``, ``substrate_a7_mps_isolated``,
    ``a7_probe_substrate_vs_integrated``) call this when ``--cuda``
    is absent / CUDA is unavailable. Without the guard, running the
    CPU-only smoke pipeline on a host that previously produced
    measurements would silently clobber the committed measured JSON
    + TeX with a stub. Returns ``True`` if the stub was written,
    ``False`` if an existing measured artifact was preserved.
    """
    json_path = pathlib.Path(json_path)
    if _is_measured(json_path) and not force:
        print(f"[safe-stub] preserving measured {json_path.name} "
              f"(status={MEASURED_STATUS}); pass --force-stub-overwrite "
              f"to clobber")
        return False
    json_path.write_text(json.dumps(stub_dict, indent=2))
    if tex_path is not None and tex_content is not None:
        pathlib.Path(tex_path).write_text(tex_content)
    return True


__all__ = (
    "cp_upper_975",
    "percentile",
    "a2_aligned_stats",
    "measure_block_transfer",
    "safe_write_stub",
    "MEASURED_STATUS",
)
