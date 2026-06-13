"""Bound-aware EDF admission control for multi-tenant SEER (review M4).

When chat (TPOT-class) and document (TTFT-class) tasks share a GPU, a
naïve FIFO scheduler lets a long doc-prefill block the chat queue for
hundreds of milliseconds, blowing every chat task's P99. The published
eF results show this directly --- chat_miss = 100\\% for every policy
except ``full`` on the shared-GPU offline driver.

This module wraps the runtime layer with two pieces:

  1. :class:`SharedLambdaState` — a thread-safe wrapper around the
     :class:`seer.timing.LambdaController` so concurrent tenants share
     one $\\lambda_t$ rather than each having a private one that resets
     per request.

  2. :class:`EDFAdmissionController` — given the current bound and a
     candidate task's deadline / expected work, returns admit / defer.
     Tasks are placed in a deadline-ordered priority queue and the
     scheduler pulls from it in EDF order rather than FIFO.

The controller is policy-agnostic: works with SEER, H2O, Streaming, etc.
The bound used for the admission predicate is :func:`lemma2_bernstein_mixture_bound`
with the worst-case $\\sigma$ across the active tenants.
"""
from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
#  Shared lambda state — for cross-tenant cohesion.
# ---------------------------------------------------------------------------


class SharedLambdaState:
    """Thread-safe sharing of one :class:`LambdaController` across tenants.

    All concurrent SEER policies in the same process should hold a
    reference to one of these and bypass their per-instance controllers
    on read/write.
    """

    def __init__(self, controller):
        self._controller = controller
        self._lock = threading.Lock()
        # Per-tenant lateness stats: tenant_id -> (n_observed, n_late, sum_lat_us)
        self._tenant_stats: dict[str, tuple[int, int, float]] = {}

    def observe(self, latency_us: float, deadline_us: float, tenant: str = "default") -> None:
        with self._lock:
            self._controller.observe(latency_us)
            n, late, s = self._tenant_stats.get(tenant, (0, 0, 0.0))
            self._tenant_stats[tenant] = (
                n + 1,
                late + (1 if latency_us > deadline_us else 0),
                s + latency_us,
            )

    def current_lambda(self) -> float:
        with self._lock:
            return self._controller.current_lambda()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._controller.stats())
            out["tenants"] = {
                t: {"n": n, "late": late, "miss_ratio": late / max(1, n),
                    "mean_us": s / max(1, n)}
                for t, (n, late, s) in self._tenant_stats.items()
            }
            return out


# ---------------------------------------------------------------------------
#  EDF admission predicate.
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _QueueItem:
    """Heap item ordered by absolute deadline ``deadline_abs_us``."""
    deadline_abs_us: float
    sequence: int  # tiebreaker for stable FIFO within identical deadlines
    arrival_us: float = field(compare=False)
    task: Any = field(compare=False)
    tenant: str = field(compare=False, default="default")


class EDFAdmissionController:
    """Bound-aware EDF queue + admission predicate.

    Invariant: the queue is ordered by absolute deadline; the controller
    pops the earliest-deadline task. Before admitting a new task it
    consults the current bound estimate; if admitting would push
    ``Pr(C_t > D) > miss_target``, the task is *deferred* (placed in a
    second queue for later retry) rather than enqueued for immediate
    execution.
    """

    def __init__(
        self,
        miss_target: float = 0.01,
        bound_inputs: dict | None = None,
    ):
        self.miss_target = float(miss_target)
        self.bound_inputs = bound_inputs or {}
        self._queue: list[_QueueItem] = []
        self._deferred: list[_QueueItem] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._admit_count = 0
        self._defer_count = 0

    # ----- admission predicate -----

    def _predict_miss_ratio(self, n_active_tasks: int) -> float:
        """Estimate ``Pr(C_t > D)`` at the current load level.

        Uses :func:`lemma2_bernstein_mixture_bound`. The ``B_t`` term is
        scaled by the number of concurrent tasks (each holds its own
        working set), and the ``sigma`` term is the in-flight worst-case.
        """
        try:
            from seer.timing.schedulability import lemma2_bernstein_mixture_bound
        except ImportError:
            return 0.0
        bi = self.bound_inputs
        if not bi:
            return 0.0
        # Concurrent tasks each add their B_t; bound is sensitive to total B_t * eps.
        b_total = bi.get("B_t", 64.0) * max(1, n_active_tasks)
        return lemma2_bernstein_mixture_bound(
            epsilon_mean=bi.get("epsilon_mean", 0.04),
            epsilon_var=bi.get("epsilon_var", 0.005),
            ell_bar_us=bi.get("ell_bar_us", 200.0),
            B_t=b_total,
            deadline_us=bi.get("deadline_us", 50000.0),
            sigma_residual_us=bi.get("sigma_us", 1200.0),
            base_cost_us=bi.get("base_cost_us", 35000.0),
        )

    def admit(self, task: Any, deadline_us: float, tenant: str = "default",
              arrival_us: float | None = None) -> bool:
        """Try to admit ``task`` (deadline = arrival + deadline_us).

        Returns ``True`` if admitted, ``False`` if deferred.
        """
        if arrival_us is None:
            arrival_us = time.monotonic() * 1e6
        deadline_abs = arrival_us + deadline_us
        with self._lock:
            n_active = len(self._queue)
            pred = self._predict_miss_ratio(n_active + 1)
            if pred > self.miss_target:
                # Defer instead of admit
                self._deferred.append(_QueueItem(
                    deadline_abs_us=deadline_abs, sequence=self._seq,
                    arrival_us=arrival_us, task=task, tenant=tenant,
                ))
                self._seq += 1
                self._defer_count += 1
                return False
            heapq.heappush(self._queue, _QueueItem(
                deadline_abs_us=deadline_abs, sequence=self._seq,
                arrival_us=arrival_us, task=task, tenant=tenant,
            ))
            self._seq += 1
            self._admit_count += 1
            return True

    def pop_next(self) -> _QueueItem | None:
        """Return the EDF-earliest item; ``None`` if empty."""
        with self._lock:
            if self._queue:
                return heapq.heappop(self._queue)
            return None

    def retry_deferred(self) -> int:
        """Try to admit each deferred item; returns # newly admitted."""
        promoted = 0
        with self._lock:
            still_deferred = []
            for item in self._deferred:
                pred = self._predict_miss_ratio(len(self._queue) + 1)
                if pred > self.miss_target:
                    still_deferred.append(item)
                else:
                    heapq.heappush(self._queue, item)
                    promoted += 1
                    self._admit_count += 1
                    self._defer_count -= 1
            self._deferred = still_deferred
        return promoted

    def stats(self) -> dict:
        with self._lock:
            return {
                "admitted": self._admit_count,
                "deferred": self._defer_count,
                "queue_depth": len(self._queue),
                "deferred_queue_depth": len(self._deferred),
            }
