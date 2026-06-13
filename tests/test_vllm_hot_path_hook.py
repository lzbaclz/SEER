"""R36e A: contract tests for the vLLM hot-path monkey-patch.

These tests use mock objects in place of vLLM's KVCacheManager and
Request, so they run on CPU only and without a vLLM import. The
intent is to pin the install/uninstall contract + the SEER router
truncation semantics so a future vLLM version bump does not
silently break the integration.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seer.integration import vllm_hot_path_hook as hph


@pytest.fixture(autouse=True)
def reset_module_state():
    """Each test starts with a clean module-level state."""
    hph._router = None
    hph._original_get_computed_blocks = None
    hph._install_count = 0
    yield
    # Cleanup any patched class after the test
    if hph._install_count > 0:
        try:
            hph.uninstall_hot_path_hook()
        except Exception:
            pass


def _make_fake_kvcm_class():
    """Build a stand-in for vllm.v1.core.kv_cache_manager.KVCacheManager."""
    class FakeKVCM:
        block_size = 32

        def get_computed_blocks(self, request):
            # The "original" behaviour: return all matched blocks.
            return request._matched_blocks, len(request._matched_blocks) * self.block_size
    return FakeKVCM


def _make_blocks(n):
    return [MagicMock(block_id=i) for i in range(n)]


def _make_request(req_id, matched_blocks):
    r = MagicMock()
    r.request_id = req_id
    r._matched_blocks = matched_blocks
    return r


def test_install_uninstall_roundtrip():
    """Install then uninstall restores the original method."""
    FakeKVCM = _make_fake_kvcm_class()
    original = FakeKVCM.get_computed_blocks
    with patch.dict(
        "sys.modules",
        {"vllm": MagicMock(),
         "vllm.v1": MagicMock(),
         "vllm.v1.core": MagicMock(),
         "vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        assert hph.is_hot_path_hook_installed()
        assert FakeKVCM.get_computed_blocks is not original
        hph.uninstall_hot_path_hook()
        assert not hph.is_hot_path_hook_installed()
        assert FakeKVCM.get_computed_blocks is original


def test_install_is_idempotent():
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        first_wrapped = FakeKVCM.get_computed_blocks
        hph.install_hot_path_hook()  # second call is no-op
        assert FakeKVCM.get_computed_blocks is first_wrapped


def test_no_router_falls_through_to_original():
    """Without a registered router, the wrapped method returns the
    original output unchanged."""
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        mgr = FakeKVCM()
        req = _make_request("r1", _make_blocks(8))
        blocks, n_tok = mgr.get_computed_blocks(req)
        assert len(blocks) == 8
        assert n_tok == 8 * mgr.block_size


def test_router_truncates_when_slack_too_small():
    """With a router that returns max_blocks_under_budget < matched, the
    hook truncates and adjusts num_tokens."""
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()

        router = MagicMock()
        # 8 matched, router says: only 3 blocks fit; prefer order [2, 5, 0, ...]
        router.partial_materialisation_order.return_value = [2, 5, 0, 1, 3, 4, 6, 7]
        router.max_blocks_under_budget.return_value = 3
        hph.register_router(router)

        mgr = FakeKVCM()
        req = _make_request("r2", _make_blocks(8))
        hph.set_request_slo_us(req, 100_000)        # 100 ms SLO
        hph.set_request_last_forward_us(req, 60_000)  # 60 ms forward → 40 ms slack
        blocks, n_tok = mgr.get_computed_blocks(req)

        assert len(blocks) == 3
        assert [b.block_id for b in blocks] == [2, 5, 0]
        assert n_tok == 3 * mgr.block_size
        router.partial_materialisation_order.assert_called_once()
        kwargs = router.partial_materialisation_order.call_args.kwargs
        assert kwargs["request_id"] == "r2"
        assert kwargs["matched_blocks"] == list(range(8))
        assert kwargs["slack_budget_us"] == pytest.approx(40_000.0)


def test_router_no_truncation_when_slack_covers_all():
    """If max_blocks_under_budget >= len(blocks), no truncation."""
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        router = MagicMock()
        router.partial_materialisation_order.return_value = list(range(4))
        router.max_blocks_under_budget.return_value = 100
        hph.register_router(router)

        mgr = FakeKVCM()
        req = _make_request("r3", _make_blocks(4))
        blocks, n_tok = mgr.get_computed_blocks(req)
        assert len(blocks) == 4
        assert n_tok == 4 * mgr.block_size


def test_zero_slack_returns_full_blocks_unchanged():
    """If slack is non-positive, hook falls through to original (vLLM
    surfaces the deadline miss)."""
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        router = MagicMock()
        hph.register_router(router)

        mgr = FakeKVCM()
        req = _make_request("r4", _make_blocks(5))
        hph.set_request_slo_us(req, 50_000)
        hph.set_request_last_forward_us(req, 60_000)  # over deadline
        blocks, n_tok = mgr.get_computed_blocks(req)

        assert len(blocks) == 5
        # Router should NOT have been called when slack <= 0
        router.partial_materialisation_order.assert_not_called()


def test_router_exception_falls_through():
    """A router bug must not crash vLLM."""
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        router = MagicMock()
        router.partial_materialisation_order.side_effect = RuntimeError("oops")
        hph.register_router(router)

        mgr = FakeKVCM()
        req = _make_request("r5", _make_blocks(4))
        hph.set_request_slo_us(req, 200_000)
        hph.set_request_last_forward_us(req, 30_000)
        blocks, n_tok = mgr.get_computed_blocks(req)
        assert len(blocks) == 4  # fell through


def test_empty_matched_blocks_skip_router():
    """If vLLM returns 0 matched blocks (cold cache), router is not called."""
    FakeKVCM = _make_fake_kvcm_class()
    with patch.dict(
        "sys.modules",
        {"vllm.v1.core.kv_cache_manager": MagicMock(KVCacheManager=FakeKVCM)},
    ):
        hph.install_hot_path_hook()
        router = MagicMock()
        hph.register_router(router)

        mgr = FakeKVCM()
        req = _make_request("r6", [])
        blocks, n_tok = mgr.get_computed_blocks(req)
        assert blocks == []
        assert n_tok == 0
        router.partial_materialisation_order.assert_not_called()


def test_uninstall_without_install_is_safe():
    hph.uninstall_hot_path_hook()  # should not raise
    assert not hph.is_hot_path_hook_installed()


def test_default_slack_uses_200ms_when_unset():
    """If set_request_slo_us is not called, the hook uses 200_000 µs default."""
    assert hph._estimate_slack_us(MagicMock(spec=[])) == 200_000.0


def test_default_last_forward_is_zero():
    """If set_request_last_forward_us is not called, hook treats forward as 0."""
    req = MagicMock(spec=[])
    hph.set_request_slo_us(req, 100_000)
    assert hph._estimate_slack_us(req) == 100_000.0
