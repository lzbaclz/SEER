"""vLLM integration entry points.

Lazy import so the rest of the seer package works without vllm
installed. Concrete connector lives in
:mod:`seer.integration.vllm_connector` and only imports vllm at
construction time.
"""
from __future__ import annotations


def is_vllm_available() -> bool:
    """Cheap probe used by the eF driver before opting into the
    vllm-backed code path."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["is_vllm_available"]
