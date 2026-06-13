"""Tests for seer.trace.datasets prompt loaders.

The eA runner defaults to ``--workload sharegpt``; before this test the
loader crashed with ``Unknown dataset: sharegpt``. We now expect the
loader to either fetch real data or fall back gracefully to RULER
synthetic when the network / dataset is gated.
"""
from __future__ import annotations

import pytest

from seer.trace.datasets import load_prompts


def test_synthetic_no_network():
    prompts = load_prompts("synthetic", [256], 4, tokenizer=None)
    assert len(prompts) == 4
    for p in prompts:
        assert "secret password" in p
        assert isinstance(p, str)


def test_ruler_alias_for_synthetic():
    prompts = load_prompts("ruler", [256], 4, tokenizer=None)
    assert len(prompts) == 4


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_prompts("not_a_dataset", [256], 4, tokenizer=None)


def test_sharegpt_falls_back_when_gated(monkeypatch):
    """If both ShareGPT mirrors are unavailable we should still get prompts."""
    import seer.trace.datasets as ds_mod

    def fake_load_dataset(*args, **kwargs):  # noqa: ARG001
        raise OSError("simulated gated dataset")

    # Simulate `from datasets import load_dataset` raising — we patch the
    # loader's `_load_sharegpt` import path. The real loader catches this
    # and falls back to RULER-synthetic.
    import datasets as real_datasets
    monkeypatch.setattr(real_datasets, "load_dataset", fake_load_dataset)

    prompts = ds_mod.load_prompts("sharegpt", [256], 3, tokenizer=None)
    assert len(prompts) == 3
    # Fallback writes RULER-style prompts that contain the marker
    assert any("secret password" in p for p in prompts)


def test_default_context_lengths_dont_explode():
    """Edge: empty context_lengths must not crash."""
    prompts = load_prompts("synthetic", [], 2, tokenizer=None)
    assert len(prompts) == 2


# ---------------------------------------------------------------------------
#  Mooncake loader
# ---------------------------------------------------------------------------

def test_mooncake_record_parses(tmp_path):
    """End-to-end: a 3-line synthetic Mooncake JSONL → 3 prompts."""
    import json

    from seer.trace.datasets import MooncakeRecord, iter_mooncake
    p = tmp_path / "trace.jsonl"
    rows = [
        {"timestamp": 0,    "input_length": 1024, "output_length": 64, "hash_ids": [1, 2, 3]},
        {"timestamp": 50,   "input_length": 2048, "output_length": 64, "hash_ids": [1, 2, 4]},
        {"timestamp": 110,  "input_length":  512, "output_length": 32, "hash_ids": [5]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))

    recs = list(iter_mooncake(source=str(p)))
    assert len(recs) == 3
    assert all(isinstance(r, MooncakeRecord) for r in recs)
    assert recs[0].input_length == 1024
    assert recs[1].hash_ids == (1, 2, 4)
    assert recs[2].timestamp_ms == 110


def test_mooncake_record_alias_fields(tmp_path):
    """Schema drift in Mooncake releases — field synonyms must be tolerated."""
    import json

    from seer.trace.datasets import iter_mooncake
    p = tmp_path / "trace.jsonl"
    rows = [
        # arxiv-era field names
        {"ts": 0, "prompt_length": 800, "completion_length": 32, "block_hashes": [10, 20]},
        # snake/different mix
        {"arrival_time_ms": 5, "prompt_tokens": 600, "completion_tokens": 16, "hash_ids": []},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    recs = list(iter_mooncake(source=str(p)))
    assert len(recs) == 2
    assert recs[0].input_length == 800
    assert recs[0].hash_ids == (10, 20)
    assert recs[1].input_length == 600
    assert recs[1].timestamp_ms == 5


def test_mooncake_load_prompts_synthesizes(tmp_path):
    """``load_prompts('mooncake', ...)`` builds prompts of the right shape."""
    import json

    from seer.trace.datasets import load_prompts
    p = tmp_path / "trace.jsonl"
    p.write_text(
        "\n".join(json.dumps({
            "timestamp": i, "input_length": 256, "output_length": 32,
            "hash_ids": [i % 5, (i + 1) % 5],
        }) for i in range(8))
    )
    import os
    os.environ["MOONCAKE_TRACE_PATH"] = str(p)
    try:
        prompts = load_prompts("mooncake", [256], 4, tokenizer=None)
    finally:
        os.environ.pop("MOONCAKE_TRACE_PATH", None)
    assert len(prompts) == 4
    # Each synthetic body contains the QA tail injected by _synthesize_prompt
    assert all("Briefly summarise" in p_ for p_ in prompts)


def test_mooncake_falls_back_when_missing(monkeypatch):
    """No env var, no cache, blocked network → fall back to RULER."""
    monkeypatch.delenv("MOONCAKE_TRACE_PATH", raising=False)
    # Block both URLs by patching the resolver
    import seer.trace.datasets as ds
    monkeypatch.setattr(ds, "_mooncake_resolve_source", lambda: None)
    prompts = ds.load_prompts("mooncake", [256], 3, tokenizer=None)
    assert len(prompts) == 3
    assert any("secret password" in p for p in prompts)


def test_mooncake_real_strict_no_fallback(monkeypatch):
    """P1-6 regression: ``mooncake-real`` MUST raise instead of
    silently falling back to RULER when the trace is unreachable.
    Paper-headline runs use this alias to guarantee the published
    numbers come from real Mooncake data."""
    monkeypatch.delenv("MOONCAKE_TRACE_PATH", raising=False)
    import seer.trace.datasets as ds
    monkeypatch.setattr(ds, "_mooncake_resolve_source", lambda: None)
    with pytest.raises(FileNotFoundError):
        ds.load_prompts("mooncake-real", [256], 3, tokenizer=None)


def test_mooncake_real_consumes_local_trace_when_available(tmp_path,
                                                            monkeypatch):
    """When ``MOONCAKE_TRACE_PATH`` points to a real JSONL, the
    ``mooncake-real`` alias must use it (no RULER fallback)."""
    import json

    import seer.trace.datasets as ds
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps({
        "timestamp": i, "input_length": 256, "output_length": 32,
        "hash_ids": [i % 4, (i + 1) % 4],
    }) for i in range(8)))
    monkeypatch.setenv("MOONCAKE_TRACE_PATH", str(p))
    prompts = ds.load_prompts("mooncake-real", [256], 4, tokenizer=None)
    assert len(prompts) == 4
    # RULER fallback would produce "secret password" — the real
    # synthesiser uses "Briefly summarise" in the QA tail.
    assert not any("secret password" in pr for pr in prompts), (
        "mooncake-real must not silently fall back to RULER synthetic"
    )


# ---------------------------------------------------------------------------
#  SQuAD-stitched Mooncake synthesiser (work3.md task I.1)
# ---------------------------------------------------------------------------

def _seed_squad_cache(tmp_path, monkeypatch, n_records=4):
    """Write a tiny synthetic SQuAD v1.1 dev file and point the resolver
    at it. Returns the (passage, question, answer) list in flatten order."""
    import json

    import seer.trace.datasets as ds
    records = [
        {"context": f"Passage number {i}. The answer is item-{i}.",
         "question": f"What is the answer for record {i}?",
         "answer": f"item-{i}"}
        for i in range(n_records)
    ]
    obj = {"data": [{"paragraphs": [{
        "context": r["context"],
        "qas": [{"question": r["question"],
                 "answers": [{"text": r["answer"], "answer_start": 0}]}],
    }]} for r in records]}
    p = tmp_path / "squad.json"
    p.write_text(json.dumps(obj))
    monkeypatch.setattr(ds, "_squad_resolve_source", lambda: str(p))
    return records


def test_squad_records_loads(monkeypatch, tmp_path):
    seeded = _seed_squad_cache(tmp_path, monkeypatch, n_records=3)
    from seer.trace.datasets import _squad_records
    recs = _squad_records()
    assert len(recs) == len(seeded)
    assert recs[0]["answer"] == "item-0"
    assert recs[2]["question"].startswith("What is the answer for record 2")


def test_squad_block_text_is_deterministic_per_hash(monkeypatch, tmp_path):
    _seed_squad_cache(tmp_path, monkeypatch, n_records=4)
    from seer.trace.datasets import _block_text_for_hash_squad, _squad_records
    recs = _squad_records()
    a = _block_text_for_hash_squad(7, recs, n_tokens=64)
    b = _block_text_for_hash_squad(7, recs, n_tokens=64)
    c = _block_text_for_hash_squad(11, recs, n_tokens=64)  # 11 % 4 == 3, 7 % 4 == 3
    d = _block_text_for_hash_squad(8, recs, n_tokens=64)   # 8  % 4 == 0
    assert a == b, "same hash must produce same text"
    assert a == c, "hashes mapping to same record must produce same text"
    assert a != d, "different records must produce different text"


def test_squad_synthesiser_preserves_prefix_share_invariant(monkeypatch, tmp_path):
    """Two requests sharing first k hash_ids must share first k blocks
    of synthesised text bit-exactly. This is the §6.1 paper claim."""
    _seed_squad_cache(tmp_path, monkeypatch, n_records=4)
    from seer.trace.datasets import (
        MooncakeRecord,
        _squad_records,
        _synthesize_prompt_squad,
    )
    records = _squad_records()
    rec_a = MooncakeRecord(
        timestamp_ms=0, input_length=4096, output_length=32,
        hash_ids=(1, 2, 3, 5),  # last hash differs from rec_b
    )
    rec_b = MooncakeRecord(
        timestamp_ms=0, input_length=4096, output_length=32,
        hash_ids=(1, 2, 3, 7),  # shares first 3 hashes with rec_a
    )
    prompt_a, ans_a = _synthesize_prompt_squad(rec_a, records, tokenizer=None)
    prompt_b, ans_b = _synthesize_prompt_squad(rec_b, records, tokenizer=None)
    # Find the first divergence: must be at or after the third "\n\n"
    # block boundary (the last hash differs).
    blocks_a = prompt_a.split("\n\n")
    blocks_b = prompt_b.split("\n\n")
    # First three blocks must match exactly.
    assert blocks_a[:3] == blocks_b[:3], "prefix-share invariant violated"
    # Reference must come from the *last* hash_id.
    assert ans_a != ans_b, "different last-hash must yield different references"


def test_load_prompts_with_refs_squad_path(monkeypatch, tmp_path):
    """End-to-end: load_prompts_with_refs('mooncake', ..., synthesiser='squad')
    returns parallel lists with non-empty references when SQuAD is seeded."""
    import json
    _seed_squad_cache(tmp_path, monkeypatch, n_records=5)
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps({
        "timestamp": i, "input_length": 256, "output_length": 32,
        "hash_ids": [i % 5, (i + 1) % 5],
    }) for i in range(6)))
    monkeypatch.setenv("MOONCAKE_TRACE_PATH", str(p))
    from seer.trace.datasets import load_prompts_with_refs
    prompts, refs = load_prompts_with_refs(
        "mooncake", [256], 4, tokenizer=None, synthesiser="squad",
    )
    assert len(prompts) == 4
    assert len(refs) == 4
    assert all(r.startswith("item-") for r in refs), \
        f"all refs should be SQuAD answers, got {refs}"


def test_load_prompts_with_refs_legacy_default(monkeypatch, tmp_path):
    """Default behaviour (synthesiser='legacy' / unset env) returns
    empty references — preserves backward compat with existing eA JSONs."""
    import json
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps({
        "timestamp": i, "input_length": 256, "output_length": 32,
        "hash_ids": [i % 5, (i + 1) % 5],
    }) for i in range(6)))
    monkeypatch.setenv("MOONCAKE_TRACE_PATH", str(p))
    monkeypatch.delenv("SEER_MOONCAKE_SYNTHESISER", raising=False)
    from seer.trace.datasets import load_prompts_with_refs
    prompts, refs = load_prompts_with_refs("mooncake", [256], 3, tokenizer=None)
    assert len(prompts) == 3
    assert refs == ["", "", ""], "legacy default must return empty refs"


def test_load_prompts_with_refs_squad_falls_back_when_no_squad(
    monkeypatch, tmp_path,
):
    """SQuAD opt-in but cache + network unavailable -> degrade to legacy
    rather than crash."""
    import json

    import seer.trace.datasets as ds
    monkeypatch.setattr(ds, "_squad_resolve_source", lambda: None)
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps({
        "timestamp": i, "input_length": 256, "output_length": 32,
        "hash_ids": [i % 5, (i + 1) % 5],
    }) for i in range(4)))
    monkeypatch.setenv("MOONCAKE_TRACE_PATH", str(p))
    prompts, refs = ds.load_prompts_with_refs(
        "mooncake", [256], 3, tokenizer=None, synthesiser="squad",
    )
    assert len(prompts) == 3
    # Legacy synthesiser → no refs
    assert refs == ["", "", ""]
