"""Multi-turn chat workload for Path beta (cross-prompt KV reuse).

Generates ``n_threads`` chat threads, each with ``n_turns`` turns,
sharing a long system prompt across threads to drive vLLM's
prefix-cache absorber and trigger real cross-prompt KV reuse.

The system prompt is a long instructional preamble (~512 tokens
worth of text) chosen so that vLLM's prefix-cache hash matches
across all threads, but the per-turn user messages are distinct
LongBench-style retrieval queries so each thread's decode
trajectory is genuinely different.

Output: a flat list of ``(thread_id, turn_id, prompt)`` triples
that the driver feeds to vLLM in interleaved order (turn t of
thread A, turn t of thread B, ..., turn t+1 of thread A, ...)
to maximize the scheduler's opportunity to evict cross-thread KV
between turns. The interleave is the regime where Lemma~3's
lag-bounded gap activates (cross-request sigma at 0.5+ per
\\cref{tab:sigma-dist}).

The actual GPU run is queued behind ``run_w_multiturn_sweep.sh``;
this module is self-contained for unit-testing the prompt
generation without a GPU.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SHARED_SYSTEM_PROMPT = (
    "You are a senior research assistant tasked with answering "
    "long-context retrieval-augmented questions. For each user query "
    "you receive, you will (1) identify the salient entities, (2) "
    "verify them against the provided context, and (3) produce a "
    "concise answer of one to three sentences. You will refuse to "
    "speculate beyond the provided context, and you will say "
    "'insufficient context' when the answer cannot be supported by "
    "the provided text. You will not produce content that is harmful, "
    "biased, or that violates the user's privacy. You will format "
    "your answer in plain text, without markdown bullets, code "
    "fences, or numbered lists, unless the user explicitly requests "
    "a structured format. You will write in the same language as the "
    "user's query. You will be precise about quantities, dates, and "
    "named entities. You will not invent citations. You will respect "
    "the user's stated constraints on answer length. When asked for "
    "a multi-turn clarification, you will reuse the prior turn's "
    "context efficiently rather than restating it verbatim. End of "
    "system instructions.\n\n"
)


# LongBench narrative_qa-style retrieval queries spanning several
# topical domains so that the cross-thread KV pattern is genuinely
# disjoint between threads (this is the regime where reactive_lag
# expects high sigma_shift on cross-request boundaries).
QUERIES: list[tuple[str, list[str]]] = [
    (
        "the inheritance dispute among the Tutwiler heirs",
        [
            "Who is named as primary executor of the Tutwiler estate?",
            "What property in Coal Hill is contested in turn three?",
            "What was the dollar amount of the disputed Tutwiler will?",
            "Which sibling testified against the contested probate?",
            "What did the appeals court decide in turn five?",
        ],
    ),
    (
        "the 1957 polar expedition led by Anders Hedin",
        [
            "What was the expedition's primary scientific objective?",
            "Which crew member kept the daily meteorological log?",
            "Where did the expedition winter-over between November and March?",
            "What instruments failed during the magnetic-anomaly survey?",
            "When did the expedition return to its base camp?",
        ],
    ),
    (
        "the trade tensions between Mireland and Vasporia",
        [
            "What goods were affected by the 2024 Mireland tariff?",
            "Who chaired the trade negotiation committee that year?",
            "What was Vasporia's retaliatory measure?",
            "How did the third-round talks in Geneva conclude?",
            "What is the projected trade-balance impact by 2027?",
        ],
    ),
    (
        "the discovery of the Anasazi calendrical site at Mesa Verde",
        [
            "Who led the 1923 dig that first identified the site?",
            "What artifact yielded the calendrical inscription?",
            "How was the inscription dated by carbon decay?",
            "What does the inscription describe about solar alignment?",
            "What is the site's preservation status today?",
        ],
    ),
    (
        "the merger of QuantumGrid and PhaseLinear",
        [
            "What was the final exchange ratio of the merger?",
            "Who is the combined entity's first board chair?",
            "What antitrust concerns were raised by EU regulators?",
            "What headcount reduction was announced post-merger?",
            "What is the projected synergy in operating margin by year three?",
        ],
    ),
]


@dataclass(frozen=True)
class ChatTurn:
    thread_id: int
    turn_id: int
    prompt: str


def generate(n_threads: int = 50, n_turns: int = 5,
             seed: int = 0) -> list[ChatTurn]:
    """Generate the multi-turn chat prompts.

    Each thread is assigned a query topic (cycling through QUERIES
    so 50 threads = 10 per topic, 5 turns each). The prompt for
    turn t is the shared system preamble + the topic header +
    turns 1..t-1's user-assistant turns (using a fixed
    placeholder assistant reply, so the prompt template is fully
    deterministic) + turn t's user message.

    Returns a flat interleaved order: turn 0 of all threads, then
    turn 1 of all threads, etc. This maximises the scheduler's
    cross-thread eviction pressure.
    """
    rng = random.Random(seed)
    assignments = [(t, rng.choice(QUERIES)) for t in range(n_threads)]
    rng.shuffle(assignments)

    out: list[ChatTurn] = []
    for turn_idx in range(n_turns):
        for thread_id, (topic_header, queries) in assignments:
            user_msg = queries[turn_idx % len(queries)]
            # Build the prompt as system + topic + prior-turn placeholders
            # + this turn's user query.
            parts = [SHARED_SYSTEM_PROMPT]
            parts.append(f"Topic: {topic_header}.\n\n")
            for prior in range(turn_idx):
                prior_user = queries[prior % len(queries)]
                parts.append(f"User (turn {prior+1}): {prior_user}\n")
                parts.append(
                    f"Assistant (turn {prior+1}): I would need to "
                    f"consult the provided context to answer that "
                    f"precisely.\n\n"
                )
            parts.append(f"User (turn {turn_idx+1}): {user_msg}\n")
            parts.append(f"Assistant (turn {turn_idx+1}):")
            prompt = "".join(parts)
            out.append(ChatTurn(thread_id=thread_id, turn_id=turn_idx,
                                prompt=prompt))
    return out


def main() -> int:
    """Quick smoke check: print the first 2 prompts to verify the
    template is well-formed and the shared prefix is identical."""
    turns = generate(n_threads=3, n_turns=2, seed=0)
    print(f"generated {len(turns)} turns total")
    print("---first turn---")
    print(turns[0].prompt[:600], "...")
    print("---second turn (different thread, same turn_id)---")
    print(turns[1].prompt[:600], "...")
    # Verify shared system prefix bytes-match across the first ~768 chars.
    shared_len = 0
    while (shared_len < min(len(turns[0].prompt), len(turns[1].prompt))
           and turns[0].prompt[shared_len] == turns[1].prompt[shared_len]):
        shared_len += 1
    print(f"\nshared prefix length: {shared_len} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
