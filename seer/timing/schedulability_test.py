"""Multi-stream schedulability test (Theorem 1 of the SEER paper).

Composes the existing Lemma 2 bound (per-task deadline-miss probability,
closed form in :mod:`seer.timing.schedulability`) with two additional
constraints to yield a feasibility verdict for a multi-stream LLM
serving task set:

  (a) Per-task bound (Lemma 2 applied to each stream i):
        eps_i(phi_i, D_i, lambda_i) <= rho_i

  (b) Shared-LAP utilization (the LAP runs once per decode step
      across the whole task set):
        sum_i lambda_i * C_LAP <= 1

  (c) Substrate-bandwidth utilization (the truncated heavy-tail bound
      keeps q_base headroom):
        sum_i lambda_i * E[C_IO_i(phi_i)] <= 1 - q_base

Schedulable(Phi) := (a) and (b) and (c).

The function returns the verdict and the binding constraint (the
violator) so an operator can read the test's output as a sizing oracle:
"infeasible because (a) on stream i" -> raise phi_i; "infeasible
because (c)" -> reduce arrival rate, switch substrate, or raise q_base
headroom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from seer.timing.schedulability import (
    _eps_eff_heuristic,
    lemma2_bernstein_mixture_bound,
    lemma2_truncated_heavy_tail_bound,
)


@dataclass(frozen=True)
class Stream:
    """One task class in the multi-stream serving task set.

    Parameters use the same units as the rest of the paper:
      lambda_rps:    arrivals per second
      deadline_us:   per-token deadline (TPOT) in microseconds
      miss_target:   per-stream miss-rate cap rho_i in [0, 1]
      phi:           operator-chosen active-mask fraction in (0, 1]
      epsilon_clean: LAP horizon-K false-negative rate at phi=1
      ell_bar_us:    substrate average miss-tier IO latency
      sigma_residual_us: sub-Gaussian residual jitter
      B_full:        per-step working-set size at phi=1 (KV blocks)
      c_io_us_at_phi: average per-step IO cost induced by phi at this
                     stream (operator computes this from substrate
                     bandwidth + (1 - phi) miss rate); used in (c).
      name:          for human-readable verdict messages
    """
    name: str
    lambda_rps: float
    deadline_us: float
    miss_target: float
    phi: float
    epsilon_clean: float
    ell_bar_us: float
    sigma_residual_us: float
    B_full: float
    c_io_us_at_phi: float


@dataclass
class Verdict:
    schedulable: bool
    binding: str = ""
    detail: dict = field(default_factory=dict)


def _eps_at_phi(stream: Stream) -> float:
    """Map the clean-regime LAP eps to the deployment-phi eps using the
    same heuristic the rest of seer.timing.schedulability uses."""
    return _eps_eff_heuristic(stream.epsilon_clean, stream.phi)


def per_stream_bound(stream: Stream, use_truncated: bool = True,
                     q_base: float = 0.0067,
                     epsilon_var_scale: float = 0.1,
                     ell_max_factor: float = 4.0) -> float:
    """Lemma 2 bound for one stream at its (phi, D, lambda).

    ``epsilon_var_scale`` is the variance proxy used to instantiate the
    Bernstein-mixture form. We default to 0.1 * eps_mean which matches
    the paper's eC table for the IO-free pool. Operators with a measured
    epsilon variance should pass their own value.
    """
    eps_mean = _eps_at_phi(stream)
    # Clamp variance into the valid Bernoulli-mixture domain
    # [0, eps_mean * (1 - eps_mean)]. At phi -> 0 the eps_eff heuristic
    # saturates to 1.0 and the maximum variance collapses to 0.
    var_max = max(0.0, eps_mean * (1.0 - eps_mean) - 1e-9)
    eps_var = min(epsilon_var_scale * eps_mean, var_max)
    common = dict(
        epsilon_mean=eps_mean,
        epsilon_var=eps_var,
        ell_bar_us=stream.ell_bar_us,
        B_t=stream.B_full * stream.phi,
        deadline_us=stream.deadline_us,
        sigma_residual_us=stream.sigma_residual_us,
        ell_max_factor=ell_max_factor,
    )
    if use_truncated:
        return lemma2_truncated_heavy_tail_bound(
            **common, escape_mass=q_base,
        )
    return lemma2_bernstein_mixture_bound(**common)


def is_schedulable(
    streams: list[Stream],
    *,
    c_lap_us: float = 33.8,
    q_base: float = 0.0067,
    use_truncated: bool = True,
    k_dec: int = 1,
) -> Verdict:
    """Theorem 1: composed schedulability test.

    c_lap_us is the LAP-side WCET budget (paper's eE result: 33.8 us
    on A100 TinyMLP TRT batch=4096). q_base is the calibrated heavy-tail
    escape mass (paper's eC q_base_holdout result: 0.0067).

    k_dec is the batched-decode amortisation factor (Corollary 1', R4-W4):
    the LAP runs once per batched decode step that fuses k_dec co-scheduled
    requests, so the (b) LHS divides by k_dec. k_dec=1 is the conservative
    sequential-server form used in the paper's main result; k_dec=2 is the
    operator-side refinement when co-scheduled batched decode is the
    deployment.
    """
    # (a) Per-task bound.
    for s in streams:
        bound = per_stream_bound(s, use_truncated=use_truncated,
                                 q_base=q_base)
        if bound > s.miss_target:
            return Verdict(
                schedulable=False,
                binding=f"(a) Lemma 2 on stream '{s.name}': "
                        f"bound {bound:.4f} > rho {s.miss_target:.4f}",
                detail={
                    "stream": s.name, "bound": bound,
                    "rho": s.miss_target, "phi": s.phi,
                },
            )

    # (b) LAP-shared utilization. C_LAP is in us; convert to seconds.
    # Corollary 1' (R4-W4): batched-decode amortisation by k_dec.
    u_lap_raw = sum(s.lambda_rps * (c_lap_us * 1e-6) for s in streams)
    u_lap = u_lap_raw / max(1, k_dec)
    if u_lap > 1.0:
        return Verdict(
            schedulable=False,
            binding=f"(b) shared LAP utilization {u_lap:.4f} > 1 "
                    f"(k_dec={k_dec})",
            detail={"u_lap": u_lap, "u_lap_raw": u_lap_raw,
                    "c_lap_us": c_lap_us, "k_dec": k_dec},
        )

    # (c) Substrate-bandwidth utilization with q_base headroom.
    u_io = sum(s.lambda_rps * (s.c_io_us_at_phi * 1e-6) for s in streams)
    capacity = 1.0 - q_base
    if u_io > capacity:
        return Verdict(
            schedulable=False,
            binding=f"(c) substrate bandwidth utilization {u_io:.4f} > "
                    f"1 - q_base = {capacity:.4f}",
            detail={"u_io": u_io, "capacity": capacity, "q_base": q_base},
        )

    return Verdict(
        schedulable=True,
        binding="",
        detail={
            "u_lap": u_lap, "u_io": u_io, "capacity": capacity,
            "per_stream_bounds": [
                {"name": s.name,
                 "bound": per_stream_bound(s, use_truncated=use_truncated,
                                           q_base=q_base),
                 "rho": s.miss_target}
                for s in streams
            ],
        },
    )
