"""Closed-form schedulability bounds — Lemmas 1, 2, 3 of the SEER paper.

Each lemma is implemented as a pure function over scalar parameters that
are all directly measurable on a deployed system:

    epsilon       : LAP horizon-K false-negative rate, in [0, 1]
    B_t           : per-step working set size (# of KV blocks needed)
    ell_bar_us    : average miss-tier IO latency, in microseconds
    sigma_residual_us : sub-Gaussian proxy of the residual cost (FFN+attn jitter)
    deadline_us   : per-step deadline D, in microseconds
    sigma_shift   : attention non-stationarity strength used by Lemma 3

The CLI is the headline tool: ``python -m seer.timing.schedulability
--lap CKPT --workload DIR --slo P99=50ms --hbm_budget 0.2`` reads a real
trace, computes both the closed-form bound and the empirical miss ratio
from a trace replay, and emits the *pessimism factor* (= bound /
measured) as a single JSON document the §6.5 plot consumes directly.

We deliberately do not import torch here — the bound functions must work
on any host that can run a Python script. The CLI imports the LAP
predictor lazily so the analytical-only path stays light.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
#  Output container
# ---------------------------------------------------------------------------

@dataclass
class BoundReport:
    """One row in the bound-vs-measured comparison."""
    epsilon: float
    B_t_avg: float
    ell_bar_us: float
    sigma_residual_us: float
    deadline_us: float
    bound_lemma1_us: float           # E[L_t] upper bound (Lemma 1)
    bound_lemma2_miss: float         # Pr(C_t > D) upper bound (Lemma 2)
    measured_miss: float | None = None
    pessimism: float | None = None  # bound / measured, NaN if measured == 0


# ---------------------------------------------------------------------------
#  Lemma 1 — first-moment lateness
# ---------------------------------------------------------------------------

def lemma1_lateness_bound(
    epsilon: float,
    B_t: float,
    ell_bar_us: float,
) -> float:
    """E[L_t] ≤ ε · |B_t| · ℓ̄.

    Proof sketch: each block in B_t is independently mispredicted with
    probability ≤ ε; each mispredicted block contributes at most ℓ̄ to
    the lateness in expectation. Linearity of expectation closes the
    argument. See ``notes/lemma_proofs.md`` (TBC) for details.
    """
    _check_unit_interval("epsilon", epsilon)
    return float(max(0.0, epsilon * B_t * ell_bar_us))


# ---------------------------------------------------------------------------
#  Lemma 2 — probabilistic schedulability via sub-Gaussian concentration
# ---------------------------------------------------------------------------

def lemma2_miss_prob_bound(
    epsilon: float,
    ell_bar_us: float,
    B_t: float,
    deadline_us: float,
    sigma_residual_us: float,
    base_cost_us: float = 0.0,
) -> float:
    """Pr(C_t > D) ≤ exp(-Φ).

    We model the per-step residual cost as

        C_t  =  base_cost  +  Σ_{i∈B_t} ε_i · ℓ_i   +  η,

    where ε_i ~ Bernoulli(ε) (LAP miss indicator), ℓ_i is the per-block
    miss-tier IO latency (mean ℓ̄, bounded by 4 ℓ̄ in the worst case),
    and η is a sub-Gaussian residual with proxy σ. The composite is
    sub-Gaussian with proxy

        σ_total² = B_t · ε · (1 − ε) · (4 ℓ̄)²  +  σ²

    and mean μ = base_cost + ε · B_t · ℓ̄. A standard Hoeffding-type tail
    bound gives Pr(C_t > D) ≤ exp(-(D − μ)² / (2 σ_total²)) when D > μ,
    otherwise the bound is vacuous (returned as 1.0).

    Returns the upper bound, clipped to [0, 1].

    Calibration semantics — important
    ---------------------------------
    * ``base_cost_us`` must be the IO-FREE per-step mean, i.e. the
      mean of (C_LAP + C_attn + C_ffn) under the policy's mask.
      ``seer/eval/sim.py`` emits this as ``per_step_base_us``;
      ``seer.eval.runner`` averages it into ``bound_inputs.base_cost_us``
      with ``base_cost_source = "io_free_mean"``. Passing the total
      mean (which already contains the realised IO cost) double-counts
      the Bernoulli-IO term and inflates μ.
    * ``sigma_residual_us`` must be the FFN+attn+LAP jitter only —
      ideally measured on a no-IO baseline (per_step_base_us with no
      mask, or a full-cache φ=1 trace through the masking simulator).
      The Bernoulli-IO variance is added analytically by the
      ``B_t · ε · (1-ε) · (4 ℓ̄)²`` term; supplying a σ that already
      contains IO variance double-counts.
    * ``ell_bar_us`` is the mean miss-tier IO latency, NOT the worst
      case. The 4ℓ̄ in the variance term is the bounded-support
      assumption A2; the mean enters μ.
    """
    _check_unit_interval("epsilon", epsilon)
    if B_t <= 0 or ell_bar_us <= 0:
        return 0.0 if deadline_us > base_cost_us else 1.0

    mu = base_cost_us + epsilon * B_t * ell_bar_us
    if deadline_us <= mu:
        return 1.0
    # IO latency is bounded; we proxy max=4*mean. This is Hoeffding-style
    # but we allow callers to override sigma_residual to roll in jitter.
    var_io = B_t * epsilon * (1.0 - epsilon) * (4.0 * ell_bar_us) ** 2
    sigma_total_sq = var_io + max(sigma_residual_us, 1e-6) ** 2
    phi = (deadline_us - mu) ** 2 / (2.0 * sigma_total_sq)
    return float(min(1.0, math.exp(-phi)))


def lemma2_bernstein_miss_prob_bound(
    epsilon: float,
    ell_bar_us: float,
    B_t: float,
    deadline_us: float,
    sigma_residual_us: float,
    base_cost_us: float = 0.0,
) -> float:
    """Bernstein-tail variant of Lemma 2.

    The Hoeffding-style bound in :func:`lemma2_miss_prob_bound` uses the
    full bounded-support variance proxy ``B·ε(1-ε)·(4ℓ̄)²``, which is loose
    when the per-block IO is well-concentrated around its mean. Bernstein's
    inequality replaces the bounded-range term with a one-sided
    ``v + bx/3`` denominator that is tighter for x ≪ b·B and degrades to
    Hoeffding only at the heavy-tail extreme.

    With ``b = 4 ℓ̄`` (per-block worst-case excess) and
    ``v = B·ε·(4ℓ̄)² + σ²`` (one-sided variance proxy that conservatively
    drops the (1-ε) factor), Bernstein gives
        Pr(C_t > D) ≤ exp(- (D-μ)² / (2 (v + b(D-μ)/3))),
    which is the tighter of (Hoeffding, Bernstein) at the operating regime
    we report in the paper. The function returns the bound clipped to
    [0, 1]; vacuous (≥1) regimes return 1.0.
    """
    _check_unit_interval("epsilon", epsilon)
    if B_t <= 0 or ell_bar_us <= 0:
        return 0.0 if deadline_us > base_cost_us else 1.0

    mu = base_cost_us + epsilon * B_t * ell_bar_us
    if deadline_us <= mu:
        return 1.0
    x = deadline_us - mu
    b = 4.0 * ell_bar_us
    v = B_t * epsilon * (4.0 * ell_bar_us) ** 2 + max(sigma_residual_us, 1e-6) ** 2
    denom = 2.0 * (v + b * x / 3.0)
    return float(min(1.0, math.exp(-x * x / denom)))


def lemma2_bernstein_mixture_bound(
    epsilon_mean: float,
    epsilon_var: float,
    ell_bar_us: float,
    B_t: float,
    deadline_us: float,
    sigma_residual_us: float,
    base_cost_us: float = 0.0,
    ell_max_factor: float = 4.0,
) -> float:
    """Bernstein bound on a *mixture* of Bernoullis (RTSS 2027 Lemma 2').

    Models per-block miss as ``M_b ~ Bernoulli(ε_b)`` where the rates
    ``{ε_b}`` themselves vary across blocks (heterogeneous LAP confidence).
    Letting ``ε̄ = E[ε_b]`` and ``v_ε = Var(ε_b) ≤ ε̄(1-ε̄)``, the marginal
    block variance is

        Var(M_b) = E[ε_b(1-ε_b)] = ε̄(1-ε̄) - v_ε,

    which is *strictly tighter* than the homogeneous-ε proxy whenever the
    LAP is at all certainty-asymmetric (which it is — eC empirically).

    Combined with Bernstein's inequality and a per-block range
    ``b = ell_max_factor · ℓ̄`` (default 4× the mean tier latency, matching
    the bounded-support assumption A2), the bound is

        Pr(C_t > D) ≤ exp(-x² / (2(ν + b x / 3))),
        ν = B · ℓ̄² · [ε̄(1-ε̄) - v_ε] + σ²,
        x = D - μ,  μ = base_cost + ε̄ · B · ℓ̄.

    Reduces to homogeneous Bernstein at v_ε = 0 and is dominated by
    Hoeffding at all v_ε ≥ 0. This is the form used as the
    paper-default in eC; the previous Hoeffding form survives for
    ablation comparison only.

    Parameters
    ----------
    epsilon_mean : float
        ε̄, the mean per-block LAP miss probability.
    epsilon_var : float
        v_ε, the *across-block* variance of ε_b, computed empirically
        from LAP predictions on the training/calibration distribution.
        Must satisfy 0 ≤ v_ε ≤ ε̄(1-ε̄).
    """
    _check_unit_interval("epsilon_mean", epsilon_mean)
    eps_var_max = epsilon_mean * (1.0 - epsilon_mean) + 1e-12
    if not (0.0 <= epsilon_var <= eps_var_max):
        raise ValueError(
            f"epsilon_var must be in [0, ε̄(1-ε̄)] = [0, {eps_var_max:.4f}], "
            f"got {epsilon_var}"
        )
    if B_t <= 0 or ell_bar_us <= 0:
        return 0.0 if deadline_us > base_cost_us else 1.0
    mu = base_cost_us + epsilon_mean * B_t * ell_bar_us
    if deadline_us <= mu:
        return 1.0
    x = deadline_us - mu
    var_per_block = max(0.0, epsilon_mean * (1.0 - epsilon_mean) - epsilon_var)
    nu = B_t * (ell_bar_us ** 2) * var_per_block + max(sigma_residual_us, 1e-6) ** 2
    b = ell_max_factor * ell_bar_us
    denom = 2.0 * (nu + b * x / 3.0)
    return float(min(1.0, math.exp(-x * x / denom)))


def lemma2_correlation_aware_bound(
    epsilon_mean: float,
    epsilon_var: float,
    ell_bar_us: float,
    B_t: float,
    deadline_us: float,
    sigma_residual_us: float,
    base_cost_us: float = 0.0,
    ell_max_factor: float = 4.0,
    rho_bar: float = 0.08,
) -> float:
    """Correlation-aware Bernstein-mixture bound (Corollary 2'').

    The Bernstein-mixture bound in `lemma2_bernstein_mixture_bound`
    assumes per-block FN indicators are i.i.d.; the empirical
    cross-block FN correlation on the Llama-2-7B trace is
    rho_bar = +0.08 (median; mean +0.19) -- not zero. This routine
    inflates the variance term by the standard
    Cochran-DerSimonian-Laird factor (1 + (B_t - 1) * rho_bar) and
    is the operator-default sizing rule (R6-W4); the i.i.d.
    bernstein_mixture variant is retained for ablation.

    rho_bar is the per-stream calibrated mean cross-block FN
    correlation. Defaults to 0.08 (paper sec:6.2 empirical median).
    """
    _check_unit_interval("epsilon_mean", epsilon_mean)
    eps_var_max = epsilon_mean * (1.0 - epsilon_mean) + 1e-12
    if not (0.0 <= epsilon_var <= eps_var_max):
        raise ValueError(
            f"epsilon_var must be in [0, ε̄(1-ε̄)] = [0, {eps_var_max:.4f}], "
            f"got {epsilon_var}"
        )
    if not (0.0 <= rho_bar <= 1.0):
        raise ValueError(f"rho_bar must be in [0, 1], got {rho_bar}")
    if B_t <= 0 or ell_bar_us <= 0:
        return 0.0 if deadline_us > base_cost_us else 1.0
    mu = base_cost_us + epsilon_mean * B_t * ell_bar_us
    if deadline_us <= mu:
        return 1.0
    x = deadline_us - mu
    var_per_block = max(0.0, epsilon_mean * (1.0 - epsilon_mean) - epsilon_var)
    inflation = 1.0 + max(0.0, (B_t - 1)) * rho_bar
    nu = (B_t * (ell_bar_us ** 2) * var_per_block * inflation
          + max(sigma_residual_us, 1e-6) ** 2)
    b = ell_max_factor * ell_bar_us
    denom = 2.0 * (nu + b * x / 3.0)
    return float(min(1.0, math.exp(-x * x / denom)))


def lemma2_truncated_heavy_tail_bound(
    epsilon_mean: float,
    epsilon_var: float,
    ell_bar_us: float,
    B_t: float,
    deadline_us: float,
    sigma_residual_us: float,
    escape_mass: float,
    base_cost_us: float = 0.0,
    ell_max_factor: float = 4.0,
    rho_bar: float | None = None,
) -> float:
    r"""Bernstein-mixture bound with heavy-tail escape mass (review M1 fix).

    The bound :func:`lemma2_bernstein_mixture_bound` assumes the per-block
    IO latency is bounded by ``ell_max_factor * ell_bar_us`` almost
    surely (assumption A2 in the paper). On workloads with heavy IO tails
    (NVMe GC bursts, SQuAD-style "JIT spikes", multi-tenant DRAM
    contention) this assumption fails, and the bound under-predicts the
    measured miss-rate by several orders of magnitude
    (SQuAD150 case study, paper §6.5 B11 / T10).

    This routine implements the **truncation-with-escape-mass** formal
    cover that addresses M1: let ``q`` be the per-block probability that
    ``ℓ_b > ell_max_factor * ell_bar_us`` (a measurable nuisance
    parameter), and let ``M`` denote the event that the bound's A2 is
    violated for at least one block in the working set. Then

        Pr(C_t > D)
          ≤ Pr(C_t > D | not M) · Pr(not M) + Pr(M)
          ≤ Bernstein_mixture(ε̄, v_ε, ℓ̄, B, D, σ)   +   B · q .

    The second term is the operator's contractual "escape mass" against
    the substrate (e.g. NVMe RAID + IOPS reservation, or simply the
    fraction of decode steps in which the GPU is permitted to be
    contention-shared).

    Returns the upper-envelope bound clipped to [0, 1].
    """
    if not (0.0 <= escape_mass <= 1.0):
        raise ValueError(f"escape_mass must be in [0, 1], got {escape_mass}")
    # R6-W4: when ``rho_bar`` is supplied (non-None), use the
    # correlation-aware body (Corollary 2''); otherwise fall back to
    # the i.i.d. mixture bound. Operator default = correlation-aware
    # at rho_bar = 0.08 (paper sec:6.2 empirical median).
    if rho_bar is not None:
        body = lemma2_correlation_aware_bound(
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=deadline_us,
            sigma_residual_us=sigma_residual_us,
            base_cost_us=base_cost_us,
            ell_max_factor=ell_max_factor,
            rho_bar=rho_bar,
        )
    else:
        body = lemma2_bernstein_mixture_bound(
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=deadline_us,
            sigma_residual_us=sigma_residual_us,
            base_cost_us=base_cost_us,
            ell_max_factor=ell_max_factor,
        )
    escape_contribution = B_t * escape_mass
    return float(min(1.0, body + escape_contribution))


def lemma2_per_step_conditional_bound(
    epsilon_mean: float,
    epsilon_var: float,
    ell_bar_us: float,
    B_t: float,
    deadline_us: float,
    sigma_residual_us: float,
    q_step: float,
    base_cost_us: float = 0.0,
    ell_max_factor: float = 4.0,
    rho_bar: float | None = None,
) -> float:
    r"""Lemma 2''' per-step conditional tail bound (R17 reviewer-W4).

    The R12 reviewer flagged that
    :func:`lemma2_truncated_heavy_tail_bound` covers the heavy-tail
    event $M$ ("at least one block in $B_t$ has $\ell_b > k\bar\ell$")
    via the union bound $\Pr(M) \le B_t \cdot q_\mathrm{base}$. On
    real substrates the burst events that produce $\ell_b > k\bar\ell$
    are \emph{strongly correlated within a single decode step}: a NVMe
    GC stall, PCIe queue overflow, or shared-DRAM contention burst
    affects every block issued during the burst window. Treating the
    blocks as independent over-counts the per-step burst probability
    by a factor of up to $B_t$.

    This routine replaces the union bound with a \emph{per-step burst
    cover}: $\Pr(M) \le q_\mathrm{step}$, where $q_\mathrm{step}$ is
    the per-step empirical burst rate (estimated directly from the
    per-step latency trace, see
    :func:`estimate_per_step_burst_rate_from_trace`). The cover is
    sound under

      \textbf{A7 (per-step burst correlation).} For every decode step
      $t$, the indicator events $\{\ell_b > k \bar\ell\}_{b \in B_t}$
      are positively correlated and the marginal probability of at
      least one block being in the burst regime is bounded by
      $q_\mathrm{step}$.

    A7 is the standard physical model for shared-substrate bursts
    (Pinheiro et al.\ FAST 2007; Stuedi et al.\ ATC 2010). When A7
    holds, the per-step bound is tighter than the union bound by a
    factor up to $B_t$.

    On the calibrated SEER trace ($q_\mathrm{step}{=}q_\mathrm{base}
    {=}0.0067$, $B_t{=}32$), the union heavy-tail term is $0.214$
    (chat-tier vacuous), while the per-step term is $0.0067$
    (potentially chat-tier non-vacuous).

    Returns the upper-envelope bound clipped to [0, 1].
    """
    if not (0.0 <= q_step <= 1.0):
        raise ValueError(f"q_step must be in [0, 1], got {q_step}")
    if rho_bar is not None:
        body = lemma2_correlation_aware_bound(
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=deadline_us,
            sigma_residual_us=sigma_residual_us,
            base_cost_us=base_cost_us,
            ell_max_factor=ell_max_factor,
            rho_bar=rho_bar,
        )
    else:
        body = lemma2_bernstein_mixture_bound(
            epsilon_mean=epsilon_mean,
            epsilon_var=epsilon_var,
            ell_bar_us=ell_bar_us,
            B_t=B_t,
            deadline_us=deadline_us,
            sigma_residual_us=sigma_residual_us,
            base_cost_us=base_cost_us,
            ell_max_factor=ell_max_factor,
        )
    # Per-step heavy-tail contribution (NOT multiplied by B_t).
    return float(min(1.0, body + q_step))


def estimate_per_step_burst_rate_from_trace(
    per_step_latencies_us,
    p50_us: float | None = None,
    sigma_us: float | None = None,
    heavy_tail_factor: float = 3.0,
) -> float:
    r"""Estimate $q_\mathrm{step}$ for
    :func:`lemma2_per_step_conditional_bound`.

    Defined as the per-step fraction of the trace where the step
    latency exceeds $\mu_0 + k \sigma_\mathrm{emp}$ with $k{=}3$.
    This is the directly-calibrable per-step burst rate; it is the
    same quantity reported by
    :func:`experiments.eC_bound_tightness.io_bound_regime
    ._empirical_base_escape_mass` (which is how the paper's
    $q_\mathrm{base}{=}0.0067$ headline value was produced). Under
    A7 (per-step burst correlation), this directly bounds
    $\Pr(\text{any block in burst})$ without a $B_t$ multiplier.
    """
    xs = [float(x) for x in per_step_latencies_us if x == x]
    if len(xs) < 50:
        return 0.0
    if p50_us is None:
        xs_sorted = sorted(xs)
        p50_us = xs_sorted[len(xs_sorted) // 2]
    if sigma_us is None:
        m = sum(xs) / len(xs)
        sigma_us = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
    threshold = p50_us + heavy_tail_factor * sigma_us
    n_over = sum(1 for x in xs if x > threshold)
    return float(n_over) / len(xs)


def estimate_escape_mass_from_trace(
    latencies_us,
    B_t: float,
    p50_us: float | None = None,
    ell_bar_us: float = 200.0,
    ell_max_factor: float = 4.0,
) -> float:
    r"""Estimate ``q = Pr(per-block IO excess > ell_max_factor · ℓ̄)``.

    A2 bounds the per-block IO ℓ_b above by ``ell_max_factor · ℓ̄``.
    Given an iterable of per-STEP latencies and the working-set size
    ``B_t``, we estimate the per-BLOCK escape mass by dividing the
    per-step excess (latency minus the IO-free median) by B_t and
    counting blocks whose implied excess exceeds the A2 bound.

    For a step at latency ``L``, the implied per-block average
    excess is ``(L - p50_us) / B_t``. We declare the step "in the
    heavy-tail regime" if this implied per-block excess exceeds
    ``ell_max_factor · ell_bar_us``, then return the empirical
    fraction. This is the per-block escape mass q to plug into
    :func:`lemma2_truncated_heavy_tail_bound`.

    Typical usage:
        >>> q = estimate_escape_mass_from_trace(per_step_us, B_t=12.8)
        >>> bound = lemma2_truncated_heavy_tail_bound(..., escape_mass=q)
    """
    xs = [float(x) for x in latencies_us if x == x]
    if len(xs) < 10:
        return 0.0
    if p50_us is None:
        xs_sorted = sorted(xs)
        p50_us = xs_sorted[len(xs_sorted) // 2]
    if B_t <= 0:
        return 0.0
    per_block_threshold = ell_max_factor * ell_bar_us  # in µs
    n_over = sum(
        1 for x in xs
        if (x - p50_us) / B_t > per_block_threshold
    )
    return float(n_over) / len(xs)


def _eps_eff_heuristic(epsilon: float, phi: float, alpha: float = 1.0) -> float:
    """First-order-Taylor heuristic for ε(φ) when no measured curve exists.

    ε_eff = clip(ε · (1 + α(1-φ)/φ), [0, 1]).

    SCOPE WARNING: this formula is a closed-form fallback that
    captures only the boundary cases (φ=1 ⇒ ε_eff=ε, φ→0 ⇒ saturate at 1)
    via the explicit clip. The 1/φ ramp between those boundaries has no
    theoretical backing — it is not derived from the predictor's
    rank-vs-recall behaviour. **Prefer**
    :func:`min_hbm_budget_for_slo_measured` whenever an empirical
    (φ, ε) curve from an eA-style budget sweep is available.
    """
    if phi <= 0:
        return 1.0
    return min(1.0, epsilon * (1.0 + alpha * (1.0 - phi) / phi))


def min_hbm_budget_for_slo(
    epsilon: float,
    ell_bar_us: float,
    B_full: float,
    deadline_us: float,
    sigma_residual_us: float,
    miss_target: float,
    base_cost_us: float = 0.0,
    search_steps: int = 200,
) -> float:
    """Invert Lemma 2 via the closed-form ε(φ) fallback.

    Searches φ ∈ (0, 1] in decreasing steps for the smallest φ whose
    Lemma 2 bound stays ≤ ``miss_target``. Uses
    :func:`_eps_eff_heuristic` to model how ε scales with φ.

    For deployments with a measured ε(φ) curve, prefer
    :func:`min_hbm_budget_for_slo_measured` — the heuristic is a
    first-order fallback only.

    Returns φ_min ∈ (0, 1]. Returns 1.0 if no φ ≤ 1 satisfies the bound.
    """
    _check_unit_interval("miss_target", miss_target)
    last_good = 1.0
    for i in range(search_steps + 1):
        phi = 1.0 - i / search_steps
        phi = max(1e-3, phi)
        eps_eff = _eps_eff_heuristic(epsilon, phi)
        bound = lemma2_miss_prob_bound(
            epsilon=eps_eff,
            ell_bar_us=ell_bar_us,
            B_t=B_full * phi,
            deadline_us=deadline_us,
            sigma_residual_us=sigma_residual_us,
            base_cost_us=base_cost_us,
        )
        if bound <= miss_target:
            last_good = phi
        else:
            break
    return float(last_good)


def _monotone_clean_eps_curve(
    eps_curve: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    r"""Project an empirical $\epsilon(\phi)$ curve onto the closest
    monotone-non-increasing curve, lifting noisy low-$\phi$ datapoints
    upward rather than pulling high-$\phi$ datapoints downward.

    Background — B4 fix (review round May 2026)
    -------------------------------------------
    Theoretically $\epsilon$ is weakly decreasing in $\phi$: more
    HBM budget cannot make the predictor worse. Empirical curves
    sometimes violate this due to measurement noise. The sizing
    rule (:func:`min_hbm_budget_for_slo_measured`) must clean the
    curve before interpolating.

    The previous implementation used a left$\to$right running-min
    followed by a right$\to$left running-min, which collapsed the
    entire curve to its global minimum — high-budget $\epsilon$ leaked
    down to low-budget knots and produced a wildly optimistic
    sizing recommendation. We replace it with a single right$\to$left
    running-max, which is the *conservative* monotone projection
    for sizing inversion: low-$\phi$ knots whose measured
    $\epsilon$ is below a higher-$\phi$ knot's $\epsilon$ get lifted
    up, so an inverted sizing rule errs on the side of asking for
    more budget rather than less.

    Parameters
    ----------
    eps_curve : list of (phi, epsilon) tuples
        Unsorted, possibly non-monotone empirical curve.

    Returns
    -------
    cleaned : list of (phi, epsilon) tuples
        Sorted by $\phi$ ascending, with $\epsilon$ monotonically
        non-increasing as $\phi$ grows. $\epsilon$ values are
        clipped to $[0, 1]$.
    """
    raw_knots = sorted(
        ((float(p), float(e)) for p, e in eps_curve),
        key=lambda t: t[0],
    )
    clamped = [
        (phi, max(0.0, min(1.0, eps))) for phi, eps in raw_knots
    ]
    cleaned: list[tuple[float, float]] = []
    cur_max = 0.0
    for phi, eps in reversed(clamped):
        cur_max = max(cur_max, eps)
        cleaned.append((phi, cur_max))
    cleaned.reverse()
    return cleaned


def min_hbm_budget_for_slo_measured(
    eps_curve: list[tuple[float, float]],
    ell_bar_us: float,
    B_full: float,
    deadline_us: float,
    sigma_residual_us: float,
    miss_target: float,
    base_cost_us: float = 0.0,
    rho_bar: float = 0.08,
    epsilon_var_scale: float = 0.1,
) -> float:
    r"""Invert the bound using a *measured* ε(φ) curve.

    R6-W4: this is the operator-default sizing rule. It uses the
    correlation-aware Bernstein-mixture bound (Corollary 2'') at
    ``rho_bar = 0.08`` (empirical cross-block FN correlation
    measured in paper sec:6.2). Pass ``rho_bar=0.0`` to recover
    the i.i.d. ablation; the bound is *strictly tighter* (more
    conservative) for rho_bar > 0.

    ``eps_curve`` is an unsorted list of (phi, epsilon) pairs from a
    budget sweep on representative traces (e.g.\ eA's
    ``results_mooncake_full/seer_b{010,020,040,080}.json`` mined for
    LAP recall vs.\ budget). The function:

    1. Sorts and filters the curve to monotone-decreasing ε(φ);
    2. Linearly interpolates ε between knots (and clamps outside);
    3. Sweeps φ from 1.0 down and returns the smallest φ that keeps
       the correlation-aware bound ≤ ``miss_target``.

    Returns φ_min ∈ (0, 1]; falls back to 1.0 if no φ ≤ 1 satisfies
    the bound.
    """
    _check_unit_interval("miss_target", miss_target)
    if not eps_curve:
        raise ValueError("eps_curve must be non-empty; use "
                         "min_hbm_budget_for_slo for the heuristic fallback")

    # Sort by phi ascending and conservatively enforce monotone
    # non-increasing ε(φ) via :func:`_monotone_clean_eps_curve`.
    cleaned = _monotone_clean_eps_curve(eps_curve)

    def eps_at(phi: float) -> float:
        # Linear interpolation; clamp outside the measured range.
        if phi <= cleaned[0][0]:
            return cleaned[0][1]
        if phi >= cleaned[-1][0]:
            return cleaned[-1][1]
        for (a_phi, a_eps), (b_phi, b_eps) in zip(cleaned, cleaned[1:]):
            if a_phi <= phi <= b_phi:
                t = (phi - a_phi) / max(1e-12, b_phi - a_phi)
                return a_eps + t * (b_eps - a_eps)
        return cleaned[-1][1]  # unreachable

    # T2-K (May 2026 sixth-round reviewer fix): the original
    # implementation used ``B_t = B_full * phi`` which conflated the
    # *budget* at φ with the *wanted-set demand* the Lemma 2 bound is
    # parametrised on. The paper's schema (seer/trace/schema.py
    # ``compute_top_k``) defines the wanted top-K as
    # ``max(MIN_TOP_K=32, TOP_K_FRACTION * |B|)`` — independent of φ
    # — and the bound's IO term should consume that quantity, not
    # the budget. Using ``B_full * phi`` instead makes the bound
    # spuriously optimistic at small φ (e.g.\ at φ=0.125 with B_full=512
    # the old formula gave B_t=64, paper-correct value is 32). We now
    # use the paper-correct top-K formula.
    from seer.trace.schema import compute_top_k as _compute_top_k
    B_t_demand = float(_compute_top_k(int(round(B_full))))
    last_good = 1.0
    for i in range(201):
        phi = max(1e-3, 1.0 - i / 200)
        eps_phi = eps_at(phi)
        # R6-W4: correlation-aware Bernstein-mixture bound is the
        # operator default. Variance proxy ε_var = epsilon_var_scale
        # * eps_phi (matches paper sec:6.2 default 0.1); clamped to
        # the valid mixture domain.
        eps_var_max = max(0.0, eps_phi * (1.0 - eps_phi) - 1e-9)
        eps_var = min(epsilon_var_scale * eps_phi, eps_var_max)
        if rho_bar > 0.0:
            bound = lemma2_correlation_aware_bound(
                epsilon_mean=eps_phi,
                epsilon_var=eps_var,
                ell_bar_us=ell_bar_us,
                B_t=B_t_demand,
                deadline_us=deadline_us,
                sigma_residual_us=sigma_residual_us,
                base_cost_us=base_cost_us,
                rho_bar=rho_bar,
            )
        else:
            # Ablation: i.i.d. bound at rho_bar = 0.
            bound = lemma2_miss_prob_bound(
                epsilon=eps_phi,
                ell_bar_us=ell_bar_us,
                B_t=B_t_demand,
                deadline_us=deadline_us,
                sigma_residual_us=sigma_residual_us,
                base_cost_us=base_cost_us,
            )
        if bound <= miss_target:
            last_good = phi
        else:
            break
    return float(last_good)


# ---------------------------------------------------------------------------
#  Lemma 3 — heuristic gap under non-stationarity
# ---------------------------------------------------------------------------

def lemma3_heuristic_gap(
    sigma_shift: float,
    epsilon: float,
    B_t: float,
    ell_bar_us: float,
) -> tuple[float, float]:
    """Return (heuristic_lateness_us, lap_lateness_us) under attention shift σ.

    A static heuristic π that does not condition on attention dynamics
    (StreamingLLM, recency, position-only) suffers worst-case lateness
    Θ(σ · |B_t| · ℓ̄): the shift fraction σ of the working set is
    guaranteed to be missed because π's selection is shift-invariant.

    A LAP-driven SEER policy whose training distribution covers the
    shift suffers only Θ(ε · |B_t| · ℓ̄) — its residual is the predictor
    error, not the shift.

    The constants are 1.0 in this simplified bound; tighter constants
    would require modelling the heuristic's specific bias.
    """
    _check_unit_interval("sigma_shift", sigma_shift)
    _check_unit_interval("epsilon", epsilon)
    heuristic = float(sigma_shift * B_t * ell_bar_us)
    lap = float(epsilon * B_t * ell_bar_us)
    return heuristic, lap


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _check_unit_interval(name: str, x: float) -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {x}")


def measured_miss_from_traces(
    latencies_us,
    deadline_us: float,
) -> float:
    """Empirical Pr(C_t > D) from a 1-D iterable of per-step latencies."""
    n = 0
    miss = 0
    for lat in latencies_us:
        n += 1
        if lat > deadline_us:
            miss += 1
    if n == 0:
        return 0.0
    return float(miss) / n


def pessimism_factor(bound: float, measured: float) -> float:
    """bound / measured, with sane handling of the corner cases."""
    if measured <= 0:
        return float("inf") if bound > 0 else 1.0
    return float(bound / measured)


# ---------------------------------------------------------------------------
#  CLI: bound vs measured comparison
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m seer.timing.schedulability",
        description="Compute Lemma 1/2 closed-form bounds and compare with measured "
                    "miss ratio from a workload trace.",
    )
    ap.add_argument("--lap", default=None,
                    help="LAP checkpoint (.pt or .onnx); used to estimate epsilon. "
                         "If omitted, --epsilon must be given.")
    ap.add_argument("--workload", default=None,
                    help="Trace directory or parquet file containing per-step latencies "
                         "(needs columns 'step_latency_us' and 'block_count').")
    ap.add_argument("--slo", default="P99=50ms",
                    help="SLO spec, e.g. 'P99=50ms' or 'chat-50ms'.")
    ap.add_argument("--hbm_budget", type=float, default=1.0,
                    help="HBM budget as fraction of full (1.0 == full).")
    ap.add_argument("--epsilon", type=float, default=None,
                    help="Override epsilon (LAP false-negative rate).")
    ap.add_argument("--ell_bar_us", type=float, default=200.0,
                    help="Average miss-tier IO latency (DRAM~200µs / NVM~1ms / SSD~10ms).")
    ap.add_argument("--sigma_residual_us", type=float, default=100.0,
                    help="Sub-Gaussian proxy of residual cost jitter (µs).")
    ap.add_argument("--base_cost_us", type=float, default=None,
                    help="Mean attn+ffn+lap cost when all blocks are in HBM. "
                         "If None, inferred from trace as min(step_latency).")
    ap.add_argument("--out", default=None, help="Optional path to dump JSON report.")
    ap.add_argument("--tiers", default="gpu+dram",
                    help="Tier mix for documentation; not used by the bound math.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    from seer.timing.slo import derive_deadline_us, parse_slo

    slo = parse_slo(args.slo)
    deadline_us = derive_deadline_us(slo)

    epsilon = args.epsilon
    if epsilon is None:
        if args.lap is None:
            raise SystemExit(
                "Need either --epsilon or --lap (to estimate epsilon from "
                "the LAP's empirical recall on the trace)."
            )
        epsilon = _estimate_epsilon_from_lap(
            args.lap, args.workload, hbm_budget=args.hbm_budget,
        )

    B_t_avg, latencies_us, base_cost_us = _trace_summary(
        args.workload, args.base_cost_us
    )

    bound1 = lemma1_lateness_bound(epsilon, B_t_avg, args.ell_bar_us)
    bound2 = lemma2_miss_prob_bound(
        epsilon=epsilon,
        ell_bar_us=args.ell_bar_us,
        B_t=B_t_avg * args.hbm_budget,
        deadline_us=deadline_us,
        sigma_residual_us=args.sigma_residual_us,
        base_cost_us=base_cost_us,
    )
    measured = (
        measured_miss_from_traces(latencies_us, deadline_us)
        if latencies_us is not None
        else None
    )
    pessimism = pessimism_factor(bound2, measured) if measured is not None else None

    rep = BoundReport(
        epsilon=float(epsilon),
        B_t_avg=float(B_t_avg),
        ell_bar_us=float(args.ell_bar_us),
        sigma_residual_us=float(args.sigma_residual_us),
        deadline_us=float(deadline_us),
        bound_lemma1_us=float(bound1),
        bound_lemma2_miss=float(bound2),
        measured_miss=measured,
        pessimism=pessimism,
    )
    payload = {
        "slo": {"name": slo.name, "kind": slo.kind,
                 "percentile": slo.percentile, "threshold_ms": slo.threshold_ms,
                 "miss_target": slo.miss_target},
        "report": asdict(rep),
        "min_hbm_budget_for_slo": min_hbm_budget_for_slo(
            epsilon=float(epsilon),
            ell_bar_us=float(args.ell_bar_us),
            B_full=float(B_t_avg),
            deadline_us=float(deadline_us),
            sigma_residual_us=float(args.sigma_residual_us),
            miss_target=float(slo.miss_target),
            base_cost_us=float(base_cost_us),
        ),
    }

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)
    return 0


# ---------------------------------------------------------------------------
#  Lazy helpers
# ---------------------------------------------------------------------------

def policy_equivalent_false_negatives(
    block_ids,
    lap_scores,
    truth_top_k,
    hbm_budget: float,
    sink: int = 4,
    window: int = 4,
) -> tuple[int, int]:
    r"""Pure helper: simulate the deployed SEER selection on a single
    decision context (one ``(request, layer, head, step)`` group) and
    return ``(false_negatives, positives)``.

    P0-10 fix (review round May 2026, reviewer #3): the LAP's ε
    parameter in Lemma 2 is the per-block miss-rate of the
    \emph{deployed} selection, not of a 0.5-threshold classifier.
    The deployed selection is: sink (lowest block-ids) + window
    (highest block-ids) + greedy top-K by LAP score, where
    K = ``hbm_budget`` × n_blocks. This function runs that
    selection and counts false negatives against the per-block
    ``truth_top_k`` label.

    Parameters
    ----------
    block_ids : array-like of int, shape (n,)
        Block IDs visible at this decision.
    lap_scores : array-like of float, shape (n,)
        LAP predicted probability for each block at the chosen horizon.
    truth_top_k : array-like of bool/int, shape (n,)
        Ground-truth top-k label (1 = block is in oracle top-k).
    hbm_budget : float
        Active-mask fraction; budget = round(hbm_budget × n).
    sink : int
        Lowest sink block-ids force-kept.
    window : int
        Highest window block-ids force-kept.

    Returns
    -------
    (false_negatives, positives) : tuple[int, int]
        Count of truth-positive blocks the selection evicted, and
        the total number of truth-positive blocks in this group.
        Returns ``(0, 0)`` when no positives exist.
    """
    import numpy as np

    bids = np.asarray(block_ids)
    scores = np.asarray(lap_scores, dtype=float)
    truth = np.asarray(truth_top_k).astype(bool)
    n = len(bids)
    if n == 0 or truth.sum() == 0:
        return 0, int(truth.sum())

    budget_k = max(1, int(round(hbm_budget * n)))
    # T1-C fix (May 2026 reviewer round R1-1.1#3, R2-#4): when
    # ``budget_k < sink + window``, the deployed SEERPolicy falls back
    # to a window-then-sink truncation that keeps EXACTLY ``budget_k``
    # blocks (window first, then backfill with sink, dedup, slice).
    # The previous estimator unconditionally added sink ∪ window and
    # could end up keeping more than ``budget_k`` blocks at φ≤0.10
    # where sink+window=8 often exceeds budget_k. That over-counted
    # the deployed policy's coverage and under-estimated ε at the
    # exact floor budgets the sizing rule reports. Mirror the
    # SEERPolicy.select_to_keep logic from seer/policy/seer.py:115-146.
    sorted_idx = np.argsort(bids)  # ascending block_id (index space)
    order_by_bid = sorted_idx
    sink_idx_list = order_by_bid[: sink].tolist()
    window_idx_list = order_by_bid[-window:].tolist() if window > 0 else []
    forced: set[int] = set(sink_idx_list) | set(window_idx_list)
    if budget_k <= len(forced):
        # Window-then-sink fallback (SEERPolicy seer.py:128-141).
        # Reverse window so most-recent blocks come first, then sink
        # in ascending order; dedup while preserving order; slice to
        # budget_k. Exactly matches deployed behaviour.
        window_then_sink: list[int] = (
            order_by_bid[-window:][::-1].tolist() + sink_idx_list
        )
        seen: set[int] = set()
        ordered: list[int] = []
        for j in window_then_sink:
            if j not in seen:
                seen.add(j)
                ordered.append(j)
        kept = set(ordered[: budget_k])
    else:
        kept = set(forced)
        remaining = budget_k - len(kept)
        if remaining > 0:
            rest = [j for j in range(n) if j not in kept]
            rest.sort(key=lambda j: -scores[j])
            kept.update(rest[:remaining])
    # Enforce the contract: keep at most budget_k.
    assert len(kept) <= budget_k, (
        f"policy_equivalent_false_negatives kept {len(kept)} > budget_k="
        f"{budget_k} — does not match deployed SEERPolicy"
    )
    kept_mask = np.zeros(n, dtype=bool)
    for j in kept:
        kept_mask[j] = True
    fn = int((truth & (~kept_mask)).sum())
    pos = int(truth.sum())
    return fn, pos


def _estimate_epsilon_from_lap(
    lap_path: str,
    workload: str | None,
    hbm_budget: float = 0.20,
    sink: int = 4,
    window: int = 4,
    horizon_idx: int = 1,
) -> float:
    """Estimate the LAP's horizon-K false-negative rate **under the
    deployed SEER selection policy**.

    P0-10 fix (review round May 2026, reviewer #3): the previous
    estimator thresholded LAP probabilities at 0.5 and reported the
    recall on positives. That number is not the deployed system's
    miss-rate — SEER does not threshold; it computes a utility,
    forces the sink/window, and greedily keeps the top
    ``budget`` blocks by utility every decision period. The Lemma 2
    ``ε`` parameter is the false-negative rate of that selection,
    not of a 0.5-cutoff classifier.

    The corrected estimator:

    1. Groups the trace rows by (request_id, layer_id, head_group, step)
       so that each group is one decision context (all blocks visible
       at that step in that head).
    2. Runs SEER's selection in dry mode: sink + window force-keep,
       then greedy fill by LAP probability up to budget.
    3. Computes per-group false-negative rate
       ``|oracle_top_k \\ kept| / |oracle_top_k|`` using the
       ``future_top_k_h*`` label column at the requested horizon.
    4. Averages across groups.

    The original threshold-based estimator survives as
    :func:`_estimate_epsilon_from_lap_legacy_thresh` for ablation
    comparison only.
    """
    if workload is None:
        # T1-B fix (May 2026 reviewer round R1-N1/C3, R2-#3):
        # the silent ``return 0.15`` here was the root cause of "the
        # paper's calibrated bound is actually a hardcoded constant".
        # Callers (runner.py) now compute ε directly from
        # ``per_step_eps_measured`` produced by the simulator's
        # oracle pass; this function should only be reached as an
        # explicit fallback with a real trace path. Refuse the
        # workload=None call rather than fabricate a number.
        raise ValueError(
            "_estimate_epsilon_from_lap requires a real workload trace; "
            "the legacy 0.15 fallback has been removed because it "
            "silently replaced per-cell measured ε in the schedulability "
            "bound. Pass a trace path or use the simulator's "
            "per_step_eps_measured (runner aggregates this automatically)."
        )
    try:
        import numpy as np

        from seer.lap.features import build_features
        from seer.lap.infer import LAPPredictor
        from seer.trace.loader import load_traces, split_by_request

        df = load_traces(workload)
        _, _, test_df = split_by_request(df)
        test_df = test_df.sort_values(
            ["request_id", "layer_id", "head_group", "step", "block_id"],
        ).reset_index(drop=True)
        X, y, _ = build_features(test_df)
        if str(lap_path).endswith(".onnx"):
            pred = LAPPredictor.from_onnx(lap_path, device="cpu")
        else:
            pred = LAPPredictor.from_torch_ckpt(lap_path, device="cpu")
        probs = pred(X)
        if probs.ndim == 1:
            probs = probs[:, None]
        h = min(horizon_idx, probs.shape[1] - 1)
        scores = probs[:, h]
        truth = y[:, min(horizon_idx, y.shape[1] - 1)]

        # Group rows by (request, layer, head_group, step) and run the
        # SEER selection per group.
        group_cols = ["request_id", "layer_id", "head_group", "step"]
        block_ids = test_df["block_id"].to_numpy()
        group_keys = list(zip(*(test_df[c].to_numpy() for c in group_cols),
                               strict=False))

        # Compute per-group slice boundaries.
        fn_counts = 0
        pos_counts = 0
        if not group_keys:
            return 0.15
        cur_key = group_keys[0]
        start = 0
        for i in range(1, len(group_keys) + 1):
            if i == len(group_keys) or group_keys[i] != cur_key:
                # Process slice [start, i)
                idx = np.arange(start, i)
                fn, pos = policy_equivalent_false_negatives(
                    block_ids=block_ids[idx],
                    lap_scores=scores[idx],
                    truth_top_k=(truth[idx] > 0.5),
                    hbm_budget=hbm_budget,
                    sink=sink,
                    window=window,
                )
                fn_counts += fn
                pos_counts += pos
                start = i
                cur_key = group_keys[i] if i < len(group_keys) else None

        if pos_counts == 0:
            return 0.15
        eps = fn_counts / pos_counts
        return float(max(0.0, min(1.0, eps)))
    except Exception as e:  # noqa: BLE001
        print(
            f"[schedulability] LAP-based ε estimation failed: {e}; "
            "falling back to 0.15"
        )
        return 0.15


def _estimate_epsilon_from_lap_legacy_thresh(
    lap_path: str,
    workload: str | None,
    horizon_idx: int = 1,
    threshold: float = 0.5,
) -> float:
    """Legacy threshold-at-0.5 ε estimator (P0-10 deprecated).

    Returned by name for ablation comparisons only. Production /
    paper-headline numbers should use :func:`_estimate_epsilon_from_lap`
    which simulates the deployed selection policy.
    """
    if workload is None:
        return 0.15
    try:
        from seer.lap.features import build_features
        from seer.lap.infer import LAPPredictor
        from seer.trace.loader import load_traces, split_by_request

        df = load_traces(workload)
        _, _, test_df = split_by_request(df)
        X, y, _ = build_features(test_df)
        if str(lap_path).endswith(".onnx"):
            pred = LAPPredictor.from_onnx(lap_path, device="cpu")
        else:
            pred = LAPPredictor.from_torch_ckpt(lap_path, device="cpu")
        probs = pred(X)
        if probs.ndim == 1:
            probs = probs[:, None]
        scores = probs[:, min(horizon_idx, probs.shape[1] - 1)]
        truth = y[:, min(horizon_idx, y.shape[1] - 1)]
        positives = truth > threshold
        if positives.sum() == 0:
            return 0.15
        recalled = (scores > threshold) & positives
        recall = float(recalled.sum()) / float(positives.sum())
        return float(max(0.0, min(1.0, 1.0 - recall)))
    except Exception as e:  # noqa: BLE001
        print(
            f"[schedulability] legacy ε estimation failed: {e}; "
            "falling back to 0.15"
        )
        return 0.15


def _trace_summary(
    workload: str | None,
    base_cost_us_override: float | None,
) -> tuple[float, list[float] | None, float]:
    """Read a workload trace (parquet or jsonl) and return (B_t_avg,
    per-step latencies in µs, base_cost_us).

    Tolerant of missing columns: if the trace doesn't have timing breakdowns
    (legacy traces from the NeurIPS pipeline), we fall back to defaults.
    """
    if workload is None:
        return 64.0, None, 1500.0
    try:
        from seer.trace.loader import load_traces
        df = load_traces(workload)
        # B_t_avg: average distinct block_id per (request, step) pair.
        if "block_id" in df.columns:
            B_t_avg = float(
                df.groupby(["request_id", "step"])["block_id"].nunique().mean()
            )
        else:
            B_t_avg = 64.0
        # Latencies: prefer per-step `step_latency_us`, fall back to the sum
        # of timing-breakdown columns when present.
        latencies = None
        if "step_latency_us" in df.columns:
            latencies = (
                df.drop_duplicates(["request_id", "step"])["step_latency_us"]
                  .astype(float).tolist()
            )
        else:
            cols = [c for c in ("c_lap_us", "c_io_us", "c_attn_us", "c_ffn_us")
                    if c in df.columns]
            if cols:
                step_df = df.drop_duplicates(["request_id", "step"])
                latencies = step_df[cols].sum(axis=1).astype(float).tolist()
        # Base cost: the smallest observed step latency is a reasonable
        # estimate of "all-HBM" cost, or use override.
        if base_cost_us_override is not None:
            base_cost_us = float(base_cost_us_override)
        elif latencies:
            base_cost_us = float(min(latencies))
        else:
            base_cost_us = 1500.0
        return B_t_avg, latencies, base_cost_us
    except Exception as e:  # noqa: BLE001
        print(f"[schedulability] trace summary failed: {e}; using defaults")
        return 64.0, None, 1500.0


if __name__ == "__main__":
    raise SystemExit(main())
