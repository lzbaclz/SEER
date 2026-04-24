"""Policy registry and factory."""
from seer.eval.policies.base import FullCachePolicy, KVPolicy
from seer.eval.policies.h2o import H2OPolicy
from seer.eval.policies.random_policy import RandomPolicy
from seer.eval.policies.recency import RecencyPolicy
from seer.eval.policies.seer_policy import SEERPolicy
from seer.eval.policies.snapkv import SnapKVPolicy
from seer.eval.policies.streaming import StreamingPolicy


def build_policy(name: str, **kwargs) -> KVPolicy:
    name = name.lower()
    if name == "full":
        return FullCachePolicy()
    if name == "streaming":
        return StreamingPolicy(**kwargs)
    if name == "h2o":
        return H2OPolicy(**kwargs)
    if name == "snapkv":
        return SnapKVPolicy(**kwargs)
    if name == "recency":
        return RecencyPolicy()
    if name == "random":
        return RandomPolicy(**kwargs)
    if name == "seer":
        return SEERPolicy(**kwargs)
    raise ValueError(f"unknown policy: {name}")


__all__ = [
    "KVPolicy",
    "FullCachePolicy",
    "StreamingPolicy",
    "H2OPolicy",
    "SnapKVPolicy",
    "RecencyPolicy",
    "RandomPolicy",
    "SEERPolicy",
    "build_policy",
]
