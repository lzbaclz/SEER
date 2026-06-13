"""Smoke test for `python -m seer.timing.schedulability` CLI.

We invoke the entry-point with no trace and a manual epsilon, and check
that it emits a JSON document containing at least the Lemma-2 bound
and the inverted ``min_hbm_budget_for_slo`` figure. No GPU / model
download required.
"""
from __future__ import annotations

import json

from seer.timing.schedulability import main as sched_main


def test_cli_no_trace_with_manual_epsilon(capsys, tmp_path):
    out = tmp_path / "rep.json"
    rc = sched_main([
        "--epsilon", "0.10",
        "--ell_bar_us", "200",
        "--sigma_residual_us", "100",
        "--slo", "P99=50ms",
        "--hbm_budget", "1.0",
        "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["report"]["epsilon"] == 0.10
    assert "bound_lemma2_miss" in payload["report"]
    assert 0.0 <= payload["report"]["bound_lemma2_miss"] <= 1.0
    assert 0.0 < payload["min_hbm_budget_for_slo"] <= 1.0

    captured = capsys.readouterr()
    assert "report" in captured.out


def test_cli_supports_preset_slo(tmp_path):
    out = tmp_path / "rep.json"
    rc = sched_main([
        "--epsilon", "0.05",
        "--slo", "chat-50ms",
        "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["slo"]["name"] == "chat-50ms"
    assert payload["slo"]["threshold_ms"] == 50.0
