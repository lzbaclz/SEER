"""Prompt loaders for benchmarks used to collect traces or drive eval.

Naming aliases (RTSS pivot):
  ``ruler`` / ``synthetic``  → both serve the RULER-style needle prompts
  ``longbench``              → LongBench v2 (one task; HuggingFace)
  ``sharegpt``               → ShareGPT public mirror (chat workload for eA)
  ``mooncake``               → Mooncake conversation trace (token-level
                                synthesis from anonymized arrival JSONL)
  ``pile``                   → The Pile 10k (legacy)

Every loader degrades to RULER-synthetic if the network fetch fails so
that smoke tests stay self-contained.
"""
from __future__ import annotations

import os
import random
from typing import Any


def load_prompts(
    dataset: str,
    context_lengths: list[int],
    num_prompts: int,
    tokenizer: Any | None = None,
) -> list[str]:
    """Return ``num_prompts`` prompts for the named workload."""
    import os as _os
    # R25 strict-workload mode: if SEER_STRICT_WORKLOAD=1 in env, OR
    # if the dataset name ends with "-strict", any silent fallback
    # to RULER-synthetic is upgraded to RuntimeError so headline
    # experiments cannot accidentally degenerate. Smoke tests run
    # without the env var and can fall back as before.
    env_strict = _os.environ.get("SEER_STRICT_WORKLOAD") in ("1", "true", "yes")
    name_raw = dataset.lower()
    if name_raw.endswith("-strict"):
        name = name_raw[: -len("-strict")]
        strict = True
    else:
        name = name_raw
        strict = env_strict

    if name in ("ruler", "synthetic"):
        if strict:
            # Synthetic IS the substrate when explicitly requested;
            # strict mode only blocks silent degradation, not
            # explicit "give me RULER" requests.
            pass
        return _load_ruler_synthetic(context_lengths, num_prompts, tokenizer)
    if name == "longbench":
        return _load_longbench(num_prompts, tokenizer, context_lengths,
                                strict_no_fallback=strict)
    if name == "sharegpt":
        return _load_sharegpt(num_prompts, tokenizer, context_lengths,
                               strict_no_fallback=strict)
    if name == "mooncake":
        # "mooncake-real" is the historical strict alias, kept for
        # back-compat; the new env-var path is the canonical one.
        return _load_mooncake(num_prompts, tokenizer, context_lengths,
                               strict_no_fallback=strict)
    if name == "mooncake-real":
        return _load_mooncake(
            num_prompts, tokenizer, context_lengths,
            strict_no_fallback=True,
        )
    if name == "pile":
        return _load_pile(num_prompts, context_lengths[0] if context_lengths else 8192,
                          tokenizer)
    raise ValueError(f"Unknown dataset: {dataset!r}. "
                     f"Known: ruler/synthetic/longbench/sharegpt/mooncake/pile "
                     f"(append '-strict' or set SEER_STRICT_WORKLOAD=1 to forbid silent fallback)")


# ---------------------------------------------------------------------------
#  Synthetic RULER: needle-in-a-haystack style
# ---------------------------------------------------------------------------

def _load_ruler_synthetic(
    context_lengths: list[int],
    num_prompts: int,
    tokenizer: Any | None,
) -> list[str]:
    if not context_lengths:
        context_lengths = [4096]
    prompts = []
    filler = "The quick brown fox jumps over the lazy dog. "
    for i in range(num_prompts):
        ctx_len = context_lengths[i % len(context_lengths)]
        repeats = max(1, ctx_len // 10)
        haystack = filler * repeats
        needle = f"\nThe secret password is {7919 * (i + 1)}.\n"
        mid = len(haystack) // 2
        prompt = haystack[:mid] + needle + haystack[mid:]
        prompt += "\n\nQ: What is the secret password? Reply with only the number.\nA:"
        if tokenizer is not None:
            ids = tokenizer.encode(prompt)[:ctx_len]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
#  LongBench
# ---------------------------------------------------------------------------

def _load_longbench(
    num_prompts: int,
    tokenizer: Any | None,
    context_lengths: list[int] | None = None,
    strict_no_fallback: bool = False,
) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", "narrativeqa", split=f"test[:{num_prompts}]")
        prompts = [f"{r['context']}\n\nQuestion: {r['input']}\nAnswer:" for r in ds]
        if tokenizer is not None and context_lengths:
            ctx = context_lengths[0]
            prompts = [_truncate(p, ctx, tokenizer) for p in prompts]
        return prompts
    except Exception as e:  # noqa: BLE001
        if strict_no_fallback:
            raise RuntimeError(
                f"LongBench unavailable under SEER_STRICT_WORKLOAD: {e!r}"
            ) from e
        print(f"[warn] LongBench unavailable, falling back to RULER synthetic: {e}")
        return _load_ruler_synthetic(context_lengths or [8192], num_prompts, tokenizer)


# ---------------------------------------------------------------------------
#  ShareGPT — primary chat workload for eA tail latency
# ---------------------------------------------------------------------------

def _load_sharegpt(
    num_prompts: int,
    tokenizer: Any | None,
    context_lengths: list[int] | None = None,
    strict_no_fallback: bool = False,
) -> list[str]:
    """Pull `num_prompts` first-turn user messages from a ShareGPT mirror.

    Tries a couple of community mirrors in order; if all fail, falls back
    to RULER-synthetic so smoke tests stay self-contained --- unless
    ``strict_no_fallback`` is set, in which case the unavailability
    raises and the caller must handle it. The strict path is what
    headline experiments use so workloads cannot silently degenerate.
    """
    candidates = [
        ("RyokoAI/ShareGPT52K", "train"),
        ("anon8231489123/ShareGPT_Vicuna_unfiltered", "train"),
    ]
    last_err: Exception | None = None
    ds = None
    for name, split in candidates:
        try:
            from datasets import load_dataset
            ds = load_dataset(name, split=split)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if ds is None:
        if strict_no_fallback:
            raise RuntimeError(
                "ShareGPT unavailable under SEER_STRICT_WORKLOAD: "
                f"{last_err!r}"
            ) from last_err
        print(f"[warn] ShareGPT unavailable, falling back to RULER synthetic: {last_err!r}")
        return _load_ruler_synthetic(context_lengths or [4096], num_prompts, tokenizer)

    rng = random.Random(0)
    indices = rng.sample(range(len(ds)), k=min(num_prompts * 4, len(ds)))
    prompts: list[str] = []
    for idx in indices:
        row = ds[idx]
        convs = row.get("conversations") or row.get("messages") or []
        first_human = next(
            (c.get("value") or c.get("content") for c in convs
             if (c.get("from") or c.get("role")) in ("human", "user")),
            None,
        )
        if not first_human or len(first_human) < 32:
            continue
        if tokenizer is not None and context_lengths:
            first_human = _truncate(first_human, context_lengths[0], tokenizer)
        prompts.append(first_human)
        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        if strict_no_fallback:
            raise RuntimeError(
                f"ShareGPT yielded only {len(prompts)}/{num_prompts} "
                "prompts under SEER_STRICT_WORKLOAD; refusing to pad "
                "with RULER")
        print(f"[warn] only collected {len(prompts)} / {num_prompts} ShareGPT prompts; "
              "padding with RULER")
        prompts.extend(
            _load_ruler_synthetic(
                context_lengths or [2048], num_prompts - len(prompts), tokenizer
            )
        )
    return prompts[:num_prompts]


# ---------------------------------------------------------------------------
#  Mooncake — production conversation trace (token-level synthesis)
# ---------------------------------------------------------------------------
#
# Mooncake (kvcache-ai/Mooncake) publishes anonymized arrival JSONL with
# the schema:
#     {"timestamp": <int_ms>, "input_length": <int_tokens>,
#      "output_length": <int_tokens>, "hash_ids": [<int>, ...]}
# No raw text — they hash 512-token blocks of the input and ship the hash
# sequence so prefix-share-aware schedulers can be benchmarked. We
# *synthesize* prompt text that:
#   1. tokenizes to exactly ``input_length`` tokens (under the caller's
#      tokenizer), and
#   2. preserves the prefix-share structure, by mapping each unique
#      ``hash_id`` to a deterministic 512-token text block. Two requests
#      sharing a hash_id at position k will share the same first
#      k×512 tokens of synthetic text.
#
# Fetch order:
#   1. ``MOONCAKE_TRACE_PATH`` env var → local JSONL file (or .gz).
#   2. ``~/.cache/seer/mooncake/trace.jsonl`` if present.
#   3. https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/.../mooncake_trace.jsonl
#      (best-effort; falls back to RULER when network unavailable).
#
# The structured ``iter_mooncake`` API (used by eF / future production
# replay) returns the parsed records, including timestamps, so callers
# can drive arrival-time-faithful experiments.

import gzip  # noqa: E402  -- mid-file imports localised to the Mooncake block
import json  # noqa: E402
import urllib.request  # noqa: E402
from collections.abc import Iterable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple  # noqa: E402


class MooncakeRecord(NamedTuple):
    """One arrival from the Mooncake anonymized trace."""
    timestamp_ms: int
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


_MOONCAKE_DEFAULT_URLS = (
    # Latest public release (subject to upstream renames).
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/"
    "mooncake_trace/mooncake_trace.jsonl",
    # Older arXiv companion release used in the Mooncake paper.
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/"
    "FAST25-release/Mooncake_Trace.jsonl",
)
_MOONCAKE_BLOCK_TOKENS = 512        # Mooncake hashes 512-token blocks
_MOONCAKE_CACHE = Path.home() / ".cache" / "seer" / "mooncake"


def _mooncake_open(path_or_url: str):
    """Open .jsonl or .jsonl.gz, local or remote, as a text stream."""
    if path_or_url.startswith(("http://", "https://")):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "seer/1.0"})
        return urllib.request.urlopen(req, timeout=30)
    p = Path(path_or_url)
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


def _mooncake_resolve_source() -> str | None:
    """Walk the env / cache / network search order, return the first hit."""
    env = os.environ.get("MOONCAKE_TRACE_PATH")
    if env and Path(env).exists():
        return env
    cached = _MOONCAKE_CACHE / "trace.jsonl"
    if cached.exists():
        return str(cached)
    for url in _MOONCAKE_DEFAULT_URLS:
        try:
            req = urllib.request.Request(url, method="HEAD",
                                          headers={"User-Agent": "seer/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return url
        except Exception:  # noqa: BLE001
            continue
    return None


def iter_mooncake(
    source: str | None = None,
    max_records: int | None = None,
) -> Iterable[MooncakeRecord]:
    """Yield :class:`MooncakeRecord` rows from the Mooncake JSONL trace.

    ``source`` overrides the auto-discovery search order. Each line is
    parsed defensively so that schema drift in upstream releases (the
    Mooncake repo has revised field names twice already) doesn't crash
    the loader — we accept multiple synonyms per field.
    """
    if source is None:
        source = _mooncake_resolve_source()
    if source is None:
        raise FileNotFoundError(
            "Mooncake trace not found. Either set MOONCAKE_TRACE_PATH, "
            "drop a copy at ~/.cache/seer/mooncake/trace.jsonl, or check "
            "network access to github."
        )
    n = 0
    with _mooncake_open(source) as fh:
        for raw in fh:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = int(rec.get("timestamp")
                     or rec.get("ts")
                     or rec.get("arrival_time_ms", 0))
            ilen = int(rec.get("input_length")
                       or rec.get("prompt_length")
                       or rec.get("prompt_tokens", 0))
            olen = int(rec.get("output_length")
                       or rec.get("completion_length")
                       or rec.get("completion_tokens", 0))
            hids = rec.get("hash_ids") or rec.get("block_hashes") or []
            yield MooncakeRecord(
                timestamp_ms=ts,
                input_length=ilen,
                output_length=olen,
                hash_ids=tuple(int(h) for h in hids),
            )
            n += 1
            if max_records is not None and n >= max_records:
                break


def _block_text_for_hash(hash_id: int, n_tokens: int = _MOONCAKE_BLOCK_TOKENS) -> str:
    """Deterministic synthetic 512-token text block keyed on hash_id.

    We re-seed a small PRNG by ``hash_id`` and emit a sentence stream of
    the requested token count (~6 chars/token rough average). The
    paragraphs are coherent English so eager-attention probing of the
    block doesn't degenerate to single-token attention sinks.
    """
    rng = random.Random(int(hash_id) & 0xFFFFFFFF)
    pool = [
        "The streaming workload exhibits roughly Zipfian access patterns over the cache.",
        "Operators size HBM by inverting the deadline-miss bound at the target percentile.",
        "Attention sinks plus a sliding window cover the deterministic part of the predictor's input.",
        "Long-tail miss-tier latency dominates the residual jitter under bursty arrivals.",
        "The PI controller's integral term clamps to avoid windup near the deadline.",
        "Each tier exposes its own bandwidth limit — DRAM, NVM, and SSD differ by an order of magnitude.",
        "We measure the predictor's WCET on locked-clock A100 silicon to remove governor noise.",
        "Hash-prefix sharing across requests amortises the prefill cost in chat sessions.",
        "Tokens occupy roughly six characters of English on average across the studied prompts.",
        "Empirical pessimism factors fall in the one-and-a-half to four times band on production traces.",
    ]
    # Emit roughly 6 chars/token; cap at desired token count after tokenize.
    target_chars = n_tokens * 6
    out: list[str] = []
    cur = 0
    while cur < target_chars:
        s = rng.choice(pool)
        out.append(s)
        cur += len(s) + 1
    return " ".join(out)


def _synthesize_prompt(
    record: MooncakeRecord,
    tokenizer: Any | None,
    fallback_ctx: int = 4096,
) -> str:
    """Build a prompt that tokenizes to exactly ``record.input_length``.

    Strategy: concatenate per-hash blocks in order, then trim/pad with
    additional generic tokens until length matches. When ``hash_ids`` is
    empty we fall back to a single ``hash_id=0`` block.
    """
    target = max(8, record.input_length or fallback_ctx)
    hids = record.hash_ids or (0,)
    body = "\n\n".join(_block_text_for_hash(h) for h in hids)
    body += (
        "\n\nQuestion: Briefly summarise the operational implications "
        "of the preceding context.\nAnswer:"
    )
    if tokenizer is not None:
        ids = tokenizer.encode(body)
        if len(ids) >= target:
            ids = ids[:target]
        else:
            # Pad with cyclic repetition of the body's tokens so the
            # final length is exact.
            pad = ids * ((target // max(1, len(ids))) + 1)
            ids = (ids + pad)[:target]
        body = tokenizer.decode(ids, skip_special_tokens=True)
    return body


def _load_mooncake(
    num_prompts: int,
    tokenizer: Any | None,
    context_lengths: list[int] | None = None,
    strict_no_fallback: bool = False,
) -> list[str]:
    """Synthesize ``num_prompts`` prompts from the Mooncake JSONL trace.

    Each prompt's token count comes from the trace's recorded
    ``input_length``; the text content is generated from per-hash_id
    block streams so prefix-share structure is preserved while staying
    license-clean.

    Parameters
    ----------
    strict_no_fallback : bool
        When True (used by the ``mooncake-real`` alias), the loader
        raises :class:`FileNotFoundError` instead of silently falling
        back to RULER when the trace cannot be reached. This is the
        flag the paper-headline runs should set so the published
        numbers cannot accidentally come from RULER synthetic data.
    """
    try:
        records = list(iter_mooncake(max_records=num_prompts * 4))
    except FileNotFoundError as e:
        if strict_no_fallback:
            raise
        print(f"[warn] Mooncake unavailable, falling back to RULER synthetic: {e}")
        return _load_ruler_synthetic(context_lengths or [4096], num_prompts, tokenizer)
    if not records:
        if strict_no_fallback:
            raise FileNotFoundError(
                "Mooncake trace resolved but produced 0 records "
                "(empty / corrupt JSONL?)"
            )
        print("[warn] Mooncake trace produced 0 records; falling back to RULER")
        return _load_ruler_synthetic(context_lengths or [4096], num_prompts, tokenizer)

    # Optionally truncate by ctx_max so we don't ingest 100k-token prompts
    # when the caller asked for ctx=4096.
    ctx_cap = (context_lengths or [None])[0]
    rng = random.Random(0)
    pool = list(records)
    rng.shuffle(pool)
    out: list[str] = []
    for rec in pool:
        if ctx_cap and rec.input_length > ctx_cap:
            rec = rec._replace(input_length=ctx_cap)
        out.append(_synthesize_prompt(rec, tokenizer, fallback_ctx=ctx_cap or 4096))
        if len(out) >= num_prompts:
            break
    if len(out) < num_prompts:
        print(f"[warn] only collected {len(out)} / {num_prompts} Mooncake prompts; "
              "padding with RULER")
        out.extend(_load_ruler_synthetic(
            context_lengths or [4096], num_prompts - len(out), tokenizer
        ))
    return out[:num_prompts]


# ---------------------------------------------------------------------------
#  Mooncake — SQuAD-stitched synthesiser (work3.md task I.1)
#
#  Replaces the random-English-block synthesiser above with one that draws
#  block content from SQuAD v1.1 passages. The hash_id -> SQuAD record
#  map is deterministic (`hash_id % len(records)`), so two requests
#  sharing the first k hash_ids still share the first k×512 tokens of
#  synthesised text bit-exact (= prefix-share invariant preserved).
#
#  In SQuAD mode, the prompt's *final* hash_id determines the question
#  and reference answer; the runner's F1 column then becomes meaningful
#  (the legacy synthesiser's F1 was ≈ 0 because the prompt content was
#  semantically degenerate).
#
#  Activation: set SEER_MOONCAKE_SYNTHESISER=squad in the environment,
#  or call load_prompts_with_refs("mooncake", ..., synthesiser="squad")
#  directly. Default behaviour is unchanged (legacy random-block) so the
#  existing committed eA / eB / eF JSONs remain reproducible from a
#  fresh checkout. To materialise the F1 improvement, see work3.md §2.1.
# ---------------------------------------------------------------------------

_SQUAD_DEFAULT_URL = (
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
)
_SQUAD_CACHE = Path.home() / ".cache" / "seer" / "squad_v1_dev.json"


def _squad_resolve_source() -> str | None:
    env = os.environ.get("SQUAD_DEV_PATH")
    if env and Path(env).exists():
        return env
    if _SQUAD_CACHE.exists():
        return str(_SQUAD_CACHE)
    return _SQUAD_DEFAULT_URL


def _squad_records() -> list[dict]:
    """Load SQuAD v1.1 dev (passage, question, answer) triples.

    Records are flattened into a deterministic order: data[i].paragraphs[j]
    .qas[k] -> one record. The hash_id -> record map (used downstream)
    is `hash_id % len(records)`, so the order is stable across runs.
    """
    src = _squad_resolve_source()
    if src is None:
        return []
    if src.startswith(("http://", "https://")):
        try:
            _SQUAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(src, headers={"User-Agent": "seer/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            _SQUAD_CACHE.write_bytes(data)
            obj = json.loads(data)
        except Exception as e:  # noqa: BLE001
            print(f"[squad] fetch failed ({e}); SQuAD synthesiser disabled")
            return []
    else:
        with open(src) as fh:
            obj = json.load(fh)
    out: list[dict] = []
    for art in obj.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = para.get("context", "")
            if not ctx:
                continue
            for qa in para.get("qas", []):
                q = qa.get("question", "")
                ans_list = qa.get("answers", [])
                if not q or not ans_list:
                    continue
                a = ans_list[0].get("text", "")
                if not a:
                    continue
                out.append({"context": ctx, "question": q, "answer": a})
    return out


def _block_text_for_hash_squad(
    hash_id: int,
    records: list[dict],
    n_tokens: int = _MOONCAKE_BLOCK_TOKENS,
) -> str:
    """Deterministic 512-token block keyed on hash_id, drawn from SQuAD.

    The block is the SQuAD passage of `records[hash_id % len(records)]`,
    cyclically padded so the rendered text reaches roughly `n_tokens`
    tokens (~6 chars/token). The question and answer of the same record
    are *not* baked into the block text — they live in the final-block
    suffix so that eviction patterns are not biased by question text.
    """
    if not records:
        return _block_text_for_hash(hash_id, n_tokens)
    rec = records[int(hash_id) % len(records)]
    body = rec["context"]
    target_chars = n_tokens * 6
    out = body
    while len(out) < target_chars:
        out = out + " " + body
    return out


def _synthesize_prompt_squad(
    record: MooncakeRecord,
    records: list[dict],
    tokenizer: Any | None,
    fallback_ctx: int = 4096,
) -> tuple[str, str]:
    """Build a SQuAD-stitched prompt + reference answer.

    Returns ``(prompt, reference)``. The prompt body is the per-hash
    SQuAD-passage stream; the suffix is the question of the *last*
    hash_id's SQuAD record (so two requests with shared prefix but
    different last block ask different questions). The reference is the
    SQuAD answer for the last hash_id's record, used by the runner's F1
    scoring.
    """
    target = max(8, record.input_length or fallback_ctx)
    hids = record.hash_ids or (0,)
    body = "\n\n".join(_block_text_for_hash_squad(h, records) for h in hids)
    last_idx = int(hids[-1]) % len(records) if records else 0
    last_rec = records[last_idx] if records else {"question": "Summarise.", "answer": ""}
    suffix = f"\n\nQuestion: {last_rec['question']}\nAnswer:"
    body = body + suffix
    if tokenizer is not None:
        ids = tokenizer.encode(body)
        # Trim from the *front* if too long, so the question suffix
        # always survives the truncation. This is the realistic prompt
        # shape for chat workloads under prefix-share scheduling.
        if len(ids) > target:
            ids = ids[-target:]
        elif len(ids) < target:
            pad = ids * ((target // max(1, len(ids))) + 1)
            ids = (pad + ids)[-target:]
        body = tokenizer.decode(ids, skip_special_tokens=True)
    return body, last_rec.get("answer", "")


def _load_mooncake_squad(
    num_prompts: int,
    tokenizer: Any | None,
    context_lengths: list[int] | None = None,
) -> tuple[list[str], list[str]]:
    """SQuAD-stitched parallel of :func:`_load_mooncake`.

    Returns ``(prompts, references)``. Falls back to the legacy
    synthesiser (with empty references) if SQuAD records cannot be
    loaded.
    """
    records = _squad_records()
    if not records:
        return _load_mooncake(num_prompts, tokenizer, context_lengths), \
               ["" for _ in range(num_prompts)]
    try:
        trace = list(iter_mooncake(max_records=num_prompts * 4))
    except FileNotFoundError as e:
        print(f"[warn] Mooncake unavailable, falling back to RULER: {e}")
        return _load_ruler_synthetic(context_lengths or [4096], num_prompts, tokenizer), \
               ["" for _ in range(num_prompts)]
    if not trace:
        return _load_ruler_synthetic(context_lengths or [4096], num_prompts, tokenizer), \
               ["" for _ in range(num_prompts)]

    ctx_cap = (context_lengths or [None])[0]
    rng = random.Random(0)
    pool = list(trace)
    rng.shuffle(pool)
    prompts: list[str] = []
    refs: list[str] = []
    for rec in pool:
        if ctx_cap and rec.input_length > ctx_cap:
            rec = rec._replace(input_length=ctx_cap)
        p, r = _synthesize_prompt_squad(
            rec, records, tokenizer, fallback_ctx=ctx_cap or 4096
        )
        prompts.append(p)
        refs.append(r)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        pad = num_prompts - len(prompts)
        prompts.extend(_load_ruler_synthetic(context_lengths or [4096], pad, tokenizer))
        refs.extend([""] * pad)
    return prompts[:num_prompts], refs[:num_prompts]


def load_prompts_with_refs(
    dataset: str,
    context_lengths: list[int],
    num_prompts: int,
    tokenizer: Any | None = None,
    synthesiser: str | None = None,
) -> tuple[list[str], list[str]]:
    """Like :func:`load_prompts` but also returns parallel reference
    answers (one per prompt; empty string when not applicable).

    For Mooncake the synthesiser is selected by, in order:
      1. the ``synthesiser`` argument (``"squad"`` / ``"legacy"``),
      2. the ``SEER_MOONCAKE_SYNTHESISER`` environment variable,
      3. the default ``"legacy"`` (preserves backward compat with the
         committed eA / eB / eF JSONs).

    Other workloads always return empty references; the runner falls
    back to its legacy ref-extraction (RULER regex / prompts_file).
    """
    name = dataset.lower()
    if name == "mooncake":
        synth = (synthesiser
                 or os.environ.get("SEER_MOONCAKE_SYNTHESISER", "legacy")).lower()
        if synth == "squad":
            return _load_mooncake_squad(num_prompts, tokenizer, context_lengths)
    prompts = load_prompts(dataset, context_lengths, num_prompts, tokenizer)
    return prompts, ["" for _ in prompts]


# ---------------------------------------------------------------------------
#  The Pile
# ---------------------------------------------------------------------------

def _load_pile(num_prompts: int, ctx_len: int, tokenizer: Any | None) -> list[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split=f"train[:{num_prompts}]")
        prompts = []
        for r in ds:
            text = r["text"]
            if tokenizer is not None:
                ids = tokenizer.encode(text)[:ctx_len]
                text = tokenizer.decode(ids, skip_special_tokens=True)
            prompts.append(text)
        return prompts
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Pile unavailable, falling back to RULER synthetic: {e}")
        return _load_ruler_synthetic([ctx_len], num_prompts, tokenizer)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_tokens: int, tokenizer: Any) -> str:
    ids = tokenizer.encode(text)[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)
