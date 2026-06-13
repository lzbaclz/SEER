"""Workload generators for the eF mixed-SLO sweep.

The default workload (chat prompts hand-coded in
``experiments.eF_mixed_slo.driver._make_chat_prompt``) is a uniform
template that differs only by request id. That workload has known
degeneracies under vLLM 0.8.5.post1:
  - Templated prefill ends up cached / fused by vLLM's batched
    prefill scheduler (the prompts share most of their tokens), so
    end-to-end latency is uniform across policies.
  - All prompts complete within a single prefill+decode batch, so
    cross-prompt KV-reuse never triggers and the scheduler's
    block_table never references worker-side save_index slots
    (the n_load=0 namespace gap documented in
    paper/sections/06_experiments.tex).

The R4 follow-through (Path beta, see ``todo_final.md``) is the
multi-turn chat workload below: 50 chat threads, each 5 turns,
shared system prompt + thread-specific user turns. With
``enable_prefix_caching=True`` this triggers real cross-prompt
KV reuse and the scheduler's allocator references real cache slot
ids (closing the namespace gap without a vLLM source patch).
"""
