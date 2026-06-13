"""SLO classes, deadline derivation, and the PI controller for adaptive lambda.

A `SLOClass` captures a production timing requirement in a form the rest of
SEER can act on. We parse strings like ``"P99=50ms"`` (per-token TPOT) or
``"TTFT-P99=1s"`` (per-request TTFT) directly from CLI flags.

`derive_deadline_us(slo, ...)` converts an SLO percentile + threshold into
the *per-step* deadline used by the schedulability analysis. For TPOT this
is just ``threshold_ms * 1000``; for TTFT we amortize across the prefill
phase using the model's measured prefill cost (callers pass ``prefill_us``).

The `LambdaController` is a textbook discrete-time PI controller. Inputs:
the rolling P99 of recent decode latencies. Output: the IO weight λ_t fed
into the SEER joint-policy utility ``p̂ − λ_t · IO_cost``. When the
observed P99 is below target the controller raises λ_t (more frugal with
IO, push quality up); when above target it drops λ_t (let go of quality,
keep deadlines).

Defaults are chosen for stability rather than aggressive tracking — RTSS
reviewers will reject a controller that chatters more than it tracks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
#  SLO classes
# ---------------------------------------------------------------------------

SLOKind = Literal["TPOT", "TTFT"]


@dataclass(frozen=True)
class SLOClass:
    """Operator-facing SLO target.

    Attributes
    ----------
    name : str
        Short tag used in result filenames (e.g. ``"chat-50ms"``).
    kind : Literal["TPOT", "TTFT"]
        Per-token (TPOT) or per-request first-token (TTFT).
    percentile : float
        Percentile of the latency distribution being constrained, in [0, 100].
        ``99.0`` means "P99".
    threshold_ms : float
        Latency target in milliseconds at the given percentile.
    miss_target : float
        Operator-acceptable miss ratio ``ρ`` — the schedulability bound
        ``Pr(C_t > D) ≤ ρ`` aims for this.
    """
    name: str
    kind: SLOKind
    percentile: float
    threshold_ms: float
    miss_target: float = 0.01  # 1% by default — matches "P99" semantics


# Common production presets. Add more here as we sweep workloads.
PRESETS: dict[str, SLOClass] = {
    "chat-25ms":   SLOClass(name="chat-25ms",   kind="TPOT", percentile=99.0, threshold_ms=25.0),
    "chat-50ms":   SLOClass(name="chat-50ms",   kind="TPOT", percentile=99.0, threshold_ms=50.0),
    "chat-100ms":  SLOClass(name="chat-100ms",  kind="TPOT", percentile=99.0, threshold_ms=100.0),
    "doc-500ms":   SLOClass(name="doc-500ms",   kind="TTFT", percentile=99.0, threshold_ms=500.0),
    "doc-1s":      SLOClass(name="doc-1s",      kind="TTFT", percentile=99.0, threshold_ms=1000.0),
    "doc-2s":      SLOClass(name="doc-2s",      kind="TTFT", percentile=99.0, threshold_ms=2000.0),
    "chat-50ms-p999": SLOClass(name="chat-50ms-p999", kind="TPOT",
                               percentile=99.9, threshold_ms=50.0, miss_target=0.001),
}

# Match `[KIND-]Pxx[xx]=NNms|NNs` with an optional "TTFT-" / "TPOT-" prefix
_SLO_RE = re.compile(
    r"""^
        (?: (?P<kind>TPOT|TTFT) [-_])?
        P(?P<pct>\d+(?:\.\d+)?)
        \s*=\s*
        (?P<val>\d+(?:\.\d+)?)
        (?P<unit>ms|s|us|µs)
    $""",
    re.VERBOSE,
)


def parse_slo(s: str) -> SLOClass:
    """Parse a string like ``"P99=50ms"`` or ``"TTFT-P99=1s"``.

    Also accepts preset names from :data:`PRESETS` (e.g. ``"chat-50ms"``).
    """
    s = s.strip()
    if s in PRESETS:
        return PRESETS[s]
    m = _SLO_RE.match(s)
    if not m:
        raise ValueError(
            f"unrecognized SLO spec: {s!r}. "
            f"Expected e.g. 'P99=50ms' or one of {sorted(PRESETS)}."
        )
    kind = m.group("kind") or "TPOT"
    pct = float(m.group("pct"))
    val = float(m.group("val"))
    unit = m.group("unit")
    val_ms = {"ms": val, "s": val * 1000.0, "us": val / 1000.0, "µs": val / 1000.0}[unit]
    miss_target = max(0.0, 1.0 - pct / 100.0)  # P99 -> 0.01, P99.9 -> 0.001
    return SLOClass(
        name=f"{kind.lower()}-P{pct:g}-{val:g}{unit}",
        kind=kind,  # type: ignore[arg-type]
        percentile=pct,
        threshold_ms=val_ms,
        miss_target=miss_target,
    )


# ---------------------------------------------------------------------------
#  Deadline derivation
# ---------------------------------------------------------------------------

def derive_deadline_us(slo: SLOClass, prefill_us: float | None = None) -> float:
    """Convert an SLO into a per-job deadline in microseconds.

    For TPOT the deadline is simply ``slo.threshold_ms`` translated to µs.
    For TTFT we amortize across the prefill phase: the operator's
    threshold covers the *whole* prefill, so the per-layer (= per-job)
    deadline is ``threshold_ms / num_prefill_steps``. When ``prefill_us``
    is the measured prefill latency at full cache, we scale by it; if
    ``None``, we treat TTFT as a single deadline (callers must do the
    amortization themselves).
    """
    if slo.kind == "TPOT":
        return float(slo.threshold_ms) * 1000.0
    # TTFT
    if prefill_us is None:
        return float(slo.threshold_ms) * 1000.0
    return max(1.0, float(slo.threshold_ms) * 1000.0 - float(prefill_us))


# ---------------------------------------------------------------------------
#  PI controller for adaptive lambda
# ---------------------------------------------------------------------------

@dataclass
class LambdaController:
    """Discrete-time PI controller mapping observed P99 TPOT → IO weight λ.

    Internal state: an integrated error term ``e_int`` and the latest
    sample ``last_err``. ``observe(latency_us)`` is called every decode
    step; ``current_lambda()`` is called by the policy at each decision
    point.

    Tuning rationale
    ----------------
    We want stability over aggressive tracking. Default ``kp=1e-4`` and
    ``ki=2e-5`` are picked so that a *single* over-deadline step does not
    immediately collapse λ to zero — instead, sustained over-deadline
    behavior (>~50 steps) drives λ down and slack-rich behavior drives it
    up. The integrator clamps prevent windup when the system is stuck
    in a regime λ cannot reach.
    """
    target_us: float
    kp: float = 1e-4
    ki: float = 2e-5
    lambda_min: float = 0.0
    lambda_max: float = 5.0
    initial: float = 0.5

    # internal state
    _lambda: float = field(default=0.5, init=False)
    _e_int: float = field(default=0.0, init=False)
    _e_last: float = field(default=0.0, init=False)
    _samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._lambda = float(self.initial)

    def reset(self) -> None:
        self._lambda = float(self.initial)
        self._e_int = 0.0
        self._e_last = 0.0
        self._samples = 0

    def observe(self, latency_us: float) -> None:
        """Feed one decode-step latency. Negative err = under target = slack.

        NaN / Inf input is dropped (fault-mode hardening, T11 / O5 review):
        a corrupted upstream measurement (e.g. CUDA event NaN, JIT-recompile
        outlier) must not contaminate the integrator. We drop, leave
        ``_lambda`` unchanged, and increment a corrupted-sample counter
        exposed via :meth:`stats`.
        """
        import math
        try:
            lat = float(latency_us)
        except (TypeError, ValueError):
            self._corrupted_count = getattr(self, "_corrupted_count", 0) + 1
            return
        if not math.isfinite(lat):
            self._corrupted_count = getattr(self, "_corrupted_count", 0) + 1
            return
        err = lat - float(self.target_us)
        # When positive (over deadline) we want lambda DOWN → use -err in update.
        # Integrator clamp avoids windup when lambda saturates.
        proposed_int = self._e_int + err
        if self.lambda_min < self._lambda < self.lambda_max:
            self._e_int = proposed_int
        # PI step: lambda decreases when err > 0 (we are over budget)
        self._lambda -= self.kp * err + self.ki * self._e_int
        # Clamp
        self._lambda = max(self.lambda_min, min(self.lambda_max, self._lambda))
        self._e_last = err
        self._samples += 1

    def current_lambda(self) -> float:
        return float(self._lambda)

    def stats(self) -> dict:
        return {
            "lambda": self._lambda,
            "e_int": self._e_int,
            "e_last": self._e_last,
            "samples": self._samples,
            "target_us": self.target_us,
            "corrupted_samples": getattr(self, "_corrupted_count", 0),
        }
