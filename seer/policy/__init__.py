"""KV-cache policy registry — RTSS-pivot home of all policies.

Migrated from :mod:`seer.eval.policies` during the RTSS pivot. The old
module path is kept as a deprecation shim so existing notebooks and
``run.sh`` scripts keep working.

Builders
--------
:func:`build_policy(name, **kwargs)` returns a :class:`KVPolicy` instance.
SEER has extra optional kwargs for SLO-aware adaptive λ and fallback;
see :class:`seer.policy.seer.SEERPolicy`.
"""
import os

from seer.policy.base import FullCachePolicy, KVPolicy
from seer.policy.baselines import (
    H2OCumulativePolicy,
    H2OPolicy,
    QuestPolicy,
    QuestUpstreamPolicy,
    RandomPolicy,
    RecencyPolicy,
    SnapKVPolicy,
    SnapKVUpstreamPolicy,
    StreamingPolicy,
)
from seer.policy.fallback import SafeFallbackPolicy
from seer.policy.infinigen import InfiniGenPolicy
from seer.policy.seer import SEERPolicy


def build_policy(name: str, **kwargs) -> KVPolicy:
    """Instantiate a policy by lowercase name.

    Recognized names (alias-tolerant)::

        full | streaming | h2o | snapkv | quest | recency | random | seer

    Unknown ``kwargs`` are forwarded to the chosen policy's ``__init__``;
    callers should consult each policy class for available options.
    """
    name = name.lower()
    if name == "full":
        return FullCachePolicy()
    if name == "streaming":
        return StreamingPolicy(**kwargs)
    if name == "h2o":
        # R31 P0-3 upstream-parity hookup: SEER_H2O_VARIANT={intree,upstream}
        # picks rolling-window heavy-hitter (in-tree H2OPolicy) vs
        # cumulative-from-step-0 (H2OCumulativePolicy, which matches
        # zhang2023h2o §3.2). Both use hh_frac=0.5 (heavy_ratio ==
        # recent_ratio of budget); the variance comes from the
        # accumulator. Without the env var the in-tree default stands.
        variant = os.environ.get("SEER_H2O_VARIANT", "intree").lower()
        # Optional knobs the parity harness sets:
        heavy_ratio = os.environ.get("SEER_H2O_HEAVY_RATIO")
        recent_ratio = os.environ.get("SEER_H2O_RECENT_RATIO")
        if heavy_ratio is not None and recent_ratio is not None:
            try:
                hr = float(heavy_ratio)
                rr = float(recent_ratio)
                total = hr + rr
                if total > 0:
                    kwargs.setdefault("hh_frac", hr / total)
            except ValueError:
                pass
        if variant == "upstream":
            return H2OCumulativePolicy(**kwargs)
        return H2OPolicy(**kwargs)
    if name == "snapkv":
        # R36d upstream-parity hookup: SEER_SNAPKV_VARIANT={intree,upstream}
        # selects prefill-frozen (in-tree) vs sliding obs-window
        # avg-pool (upstream FasterDecoding/SnapKV §3.2). Without the
        # env var the in-tree default stands.
        variant = os.environ.get("SEER_SNAPKV_VARIANT", "intree").lower()
        obs_window = os.environ.get("SEER_SNAPKV_OBS_WINDOW")
        if obs_window is not None:
            try:
                kwargs.setdefault("obs_window", int(obs_window))
            except ValueError:
                pass
        if variant == "upstream":
            return SnapKVUpstreamPolicy(**kwargs)
        # in-tree SnapKVPolicy doesn't take obs_window; drop it
        kwargs.pop("obs_window", None)
        return SnapKVPolicy(**kwargs)
    if name == "quest":
        # R36d upstream-parity hookup: SEER_QUEST_VARIANT={intree,upstream}
        # selects one-block-per-page (in-tree) vs sub-page max-attention
        # (upstream Tang et al. ICML 2024). Upstream recent_floor=8.
        variant = os.environ.get("SEER_QUEST_VARIANT", "intree").lower()
        if variant == "upstream":
            kwargs.setdefault("recent_floor", 8)
            return QuestUpstreamPolicy(**kwargs)
        return QuestPolicy(**kwargs)
    if name == "infinigen":
        return InfiniGenPolicy(**kwargs)
    if name == "recency":
        return RecencyPolicy(**kwargs)
    if name == "random":
        return RandomPolicy(**kwargs)
    if name == "seer":
        return SEERPolicy(**kwargs)
    if name == "seer-warmup":
        from seer.policy.speculative_warmup import SpeculativeWarmupPolicy
        # Pop warmup-specific kwargs; remainder go to the wrapped SEERPolicy.
        warmup_kwargs = {}
        for k in ("trigger_threshold", "trigger_miss_rate", "miss_window",
                  "warmup_size", "cooldown_steps"):
            if k in kwargs:
                warmup_kwargs[k] = kwargs.pop(k)
        primary = SEERPolicy(**kwargs)
        return SpeculativeWarmupPolicy(primary=primary, **warmup_kwargs)
    raise ValueError(f"unknown policy: {name}")


__all__ = [
    "KVPolicy",
    "FullCachePolicy",
    "StreamingPolicy",
    "H2OPolicy",
    "H2OCumulativePolicy",
    "SnapKVPolicy",
    "SnapKVUpstreamPolicy",
    "QuestPolicy",
    "QuestUpstreamPolicy",
    "InfiniGenPolicy",
    "RecencyPolicy",
    "RandomPolicy",
    "SEERPolicy",
    "SafeFallbackPolicy",
    "build_policy",
]
