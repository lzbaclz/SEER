# R36d Operator runbook: A7 under MIG hard partitioning

**Purpose.** Close reviewer R36's #1 RTSS-push item:
"做一个 'pass A7 的真实隔离部署' 完整闭环：MIG 或 cgroup-PCIe,
substrate + integrated + hot-path probe 全部 HOLDS, 然后用它跑
sizing verdict."

This is a **human-readable runbook**; do not `bash` it directly.
The executable companion is [`scripts/run_mig_a7.sh`](../scripts/run_mig_a7.sh).

## One-line run (default: 3g.40gb on GPU 1)

```bash
sudo -E bash scripts/run_mig_a7.sh
```

That's it. The script does the six steps below, and on completion
the paper claim-evidence table + manifest entry 33 are ready to be
re-rendered with `python scripts/gen_claim_evidence_table.py && make paper`.

## Knobs (set via env-vars to override defaults)

| Env-var       | Default | Meaning |
| ------------- | ------- | ------- |
| `TARGET_GPU`  | `1`     | Physical GPU index to MIG-enable |
| `PROFILE`     | `9`     | MIG profile (9 = 3g.40gb half, 19 = 1g.10gb tightest) |
| `CONTENDER_MB`| `1.0`   | Co-tenant buffer size |
| `N_REPS`      | `5000`  | Samples per block size |
| `OUT_STEM`    | `substrate_a7_mig` | Output filename stem |
| `KEEP_MIG`    | `0`     | `1` = leave MIG enabled (for cross-partition follow-up) |
| `CONDA_SH`    | `/home/lzq/miniconda3/etc/profile.d/conda.sh` | conda activate path |

Example — tightest single-partition test:

```bash
TARGET_GPU=0 PROFILE=19 CONTENDER_MB=1.0 \
  sudo -E bash scripts/run_mig_a7.sh
```

## What the script does (the six steps)

1. **Sanity check**: confirms `TARGET_GPU` has no active CUDA processes.
2. **Enable MIG**: `nvidia-smi -i $TARGET_GPU -mig 1`.
3. **Create Compute Instance**: `nvidia-smi mig -i $TARGET_GPU -cgi $PROFILE -C`.
4. **Read MIG UUID**: parses `nvidia-smi -L`.
5. **Run A7 probe** under the MIG partition (`CUDA_VISIBLE_DEVICES=MIG-<uuid>`).
6. **Cleanup**: destroys CI/GI and disables MIG (unless `KEEP_MIG=1`).

## Output

* `experiments/eC_bound_tightness/results/substrate_a7_mig.json` — measured
* `experiments/eC_bound_tightness/results/substrate_a7_mig.tex` — paper-ready
* Stdout final verdict: `HOLDS` or `FAILS` on every block under MIG + contender.

## Expected verdicts (both publishable)

* **A7 HOLDS on MIG 3g.40gb + 1 MiB intra-partition contender** —
  closes reviewer's #1 ask; SEER deployable on hard-partitioned MIG
  instances. The §VI.F A7 pass/fail line gains a fifth `HOLD` row.
* **A7 FAILS even on MIG 1g.10gb** — MIG bounds SM share but not
  PCIe bandwidth on this platform; motivates cgroup-PCIe (queued
  `todo_atc.md`~D). The disclaimer in §VI.F's MPS-deployment-ceiling
  paragraph picks up the measured number.

## Cross-partition isolation (optional, tighter test)

The default runs both probe and contender on the *same* MIG
partition. For the stronger "MIG isolates cross-tenant interference"
claim, create two GI+CI on the same physical GPU and put the
contender on a different one:

```bash
# Step A: enable MIG and create two 3g.40gb instances
TARGET_GPU=1 PROFILE=9 KEEP_MIG=1 sudo -E bash scripts/run_mig_a7.sh
sudo nvidia-smi mig -i 1 -cgi 9 -C   # second GI+CI on same GPU

# Step B: capture both UUIDs and rerun probe on partition A with
#         contender on partition B
MIG_A=$(nvidia-smi -L | awk -F'UUID: ' '/MIG-/{print $2}' | tr -d ')' | sed -n '1p')
MIG_B=$(nvidia-smi -L | awk -F'UUID: ' '/MIG-/{print $2}' | tr -d ')' | sed -n '2p')
source /home/lzq/miniconda3/etc/profile.d/conda.sh && conda activate seer
CUDA_VISIBLE_DEVICES=$MIG_A \
    SEER_MIG_CONTENDER_UUID=$MIG_B \
    python -m experiments.eC_bound_tightness.substrate_a7_mig \
      --cuda --device-index 0 --contender-mb 1.0 \
      --n-reps 5000 --out-stem substrate_a7_mig_cross

# Step C: cleanup
sudo nvidia-smi mig -i 1 -dci ; sudo nvidia-smi mig -i 1 -dgi
sudo nvidia-smi -i 1 -mig 0
```

Output: `results/substrate_a7_mig_cross.{json,tex}`.

## After running

```bash
python scripts/gen_claim_evidence_table.py    # picks up the new JSON
make paper                                    # rebuilds paper/main.pdf
git add -A paper/ experiments/eC_bound_tightness/results/substrate_a7_mig*
git commit -m "R36d-mig: measured A7 verdict under MIG hard partitioning"
```
