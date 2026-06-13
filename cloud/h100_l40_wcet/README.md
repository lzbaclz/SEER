# H100 / L40 Cross-Hardware LAP WCET Kit

This is the self-contained kit you bring up on a rented cloud GPU
(RunPod / Lambda / GCP / Vast.ai) to fill in the H100 and L40 rows of
the §6.7 (eE) WCET table. The local repo ships only A100 numbers; the
RTSS reviewer asked whether the 200 µs LAP WCET budget holds on
Hopper / Ada / lower-SM-count SKUs. This kit answers that.

> **For the full clone-on-cloud → push → pull-on-laptop workflow,
> read [WORKFLOW.md](WORKFLOW.md).** It walks through SSH key setup,
> shipping the LAP checkpoints (which are git-ignored, so `git clone`
> alone is not enough), running the sweep, and committing back via
> `git add -f` (the `results/` directory is also git-ignored).

Total cloud spend per SKU: **≈ $1–3 in compute** (5 minutes on a single
GPU). Plan to budget **30 min wall-clock per SKU** including environment
setup.

## TL;DR — minimum command list

On your laptop:

```bash
cd ~/codes/SEER && git push origin main
tar czf /tmp/seer_checkpoints.tgz checkpoints/lap_prod_*.{plan,onnx} \
                                  checkpoints/lap_prod_{tiny,rnn,xfmr}.pt
scp /tmp/seer_checkpoints.tgz root@<cloud-ip>:/tmp/
```

On the cloud node:

```bash
cd /workspace
git clone https://<TOKEN>@github.com/lzbaclz/SEER.git    # or SSH
cd SEER
tar xzf /tmp/seer_checkpoints.tgz
git config user.email "1145885122@qq.com" && git config user.name "lzbaclz"

bash cloud/h100_l40_wcet/setup_env.sh              # ~3 min, automatic
source .venv-cloud/bin/activate
sudo bash cloud/h100_l40_wcet/lock_clocks.sh       # optional, root only
GPU_INDEX=0 bash cloud/h100_l40_wcet/run.sh        # ~5 min
python cloud/h100_l40_wcet/verify.py --results_dir experiments/eE_lap_wcet/results

GPU_TAG=$(ls experiments/eE_lap_wcet/results/lap_wcet_*.json \
            | grep -v _A100_ | head -1 | sed 's|.*/lap_wcet_||; s|_tiny_mlp.*||')
git add -f experiments/eE_lap_wcet/results/lap_wcet_*${GPU_TAG}*.json \
           experiments/eE_lap_wcet/results/_run_summary_${GPU_TAG}.txt
git commit -m "eE: ${GPU_TAG} cross-hardware WCET"
git push origin main
```

Back on your laptop:

```bash
cd ~/codes/SEER && git pull origin main
bash cloud/h100_l40_wcet/integrate.sh              # rebuilds figure + emits paper patch
# paste cloud/h100_l40_wcet/paper_patches.txt into the indicated paper sections
```

That's the whole loop. **Yes, the environment is automatic** —
`setup_env.sh` installs PyTorch / ONNX-Runtime / TensorRT 10 /
SEER end-to-end on a bare CUDA 12.1+ image. The two manual things
are (a) SSH'ing the checkpoint tarball over (because checkpoints
are git-ignored) and (b) `git add -f` on the produced JSONs
(because `results/` is also git-ignored). Both are spelled out in
[WORKFLOW.md](WORKFLOW.md).

## What this folder contains

| File | Purpose |
| --- | --- |
| `setup_env.sh` | Bootstrap a fresh CUDA-12.1+ cloud node with PyTorch 2.6, ONNX-Runtime, and TensorRT 10. Idempotent. |
| `run.sh` | Sweeps the eE grid (3 archs × 4 backends × 4 batch sizes), tagged with the GPU's `nvidia-smi --query-gpu=name`. Self-correcting CLI that matches `seer.lap.wcet`'s actual flags. |
| `lock_clocks.sh` | Auto-detects max SM clock for the present SKU and locks it. Required for stable WCET numbers. |
| `verify.py` | Sanity-check the JSON outputs, print one-line "P50 / P99 / P99.9 vs 200 µs budget" report. |
| `integrate.sh` | Copy results back into `experiments/eE_lap_wcet/results/` and re-render the eE figure. |
| `expected.json` | A100 reference numbers — `verify.py` compares fresh runs against this. |

## Step-by-step

### 1. Pick a node

| GPU | Memory | Suggested provider | $/hour (spot) |
| --- | --- | --- | --- |
| H100 SXM5 80 GB | 80 GB | RunPod Spot, Lambda On-Demand | ≈ $2.50 |
| H100 PCIe 80 GB | 80 GB | RunPod Community Cloud | ≈ $1.99 |
| L40 / L40S | 48 GB | RunPod Spot, Vast.ai | ≈ $1.10 |

You only need a single GPU. Do **not** rent multi-GPU instances —
WCET measurement is single-card by construction (NVLink probe is in
a separate eE sub-experiment).

### 2. Bring up the environment

```bash
# On the cloud node, after SSH-in:
git clone https://github.com/<your-repo>/SEER.git
cd SEER
bash cloud/h100_l40_wcet/setup_env.sh    # ~3-5 min
```

The setup script:
* Installs PyTorch 2.6 with CUDA 12.1 wheels.
* Installs ONNX-Runtime GPU.
* Installs TensorRT 10.x wheels (NVIDIA's PyPI index).
* Installs the SEER package (`pip install -e .`).
* Verifies `python -c "import torch; assert torch.cuda.is_available()"`.

If any step fails, the script aborts with a one-line diagnostic
(it will tell you to e.g. install a newer driver). Fix it and re-run
— the script is idempotent.

### 3. Lock SM clocks (recommended, requires sudo / root)

```bash
sudo bash cloud/h100_l40_wcet/lock_clocks.sh
```

This queries `nvidia-smi -q -d SUPPORTED_CLOCKS` and locks the SM clock
to the **second-highest** supported value (avoids hot-throttling at the
absolute max while still giving determinism). On H100 this is 1410–1980
MHz depending on PCIe vs SXM5; on L40 ≈ 2520 MHz.

If your cloud provider doesn't grant root (e.g. RunPod's Community
tier), skip this step — the WCET numbers will be ≈ 5 % more variable
but the headline P99.9 should still beat the 200 µs budget.

### 4. Run the sweep

```bash
GPU_INDEX=0 bash cloud/h100_l40_wcet/run.sh
```

Output goes to `experiments/eE_lap_wcet/results/lap_wcet_*_<GPU_NAME>.json`
(one file per `arch × backend × batch` combination). Filenames carry
the GPU SKU as a suffix so they don't collide with the A100 results
already in the repo.

The grid (default):
* archs: `tiny_mlp`, `block_rnn`, `block_xfmr`
* backends: `tensorrt-fp32`, `tensorrt-fp16`, `onnx`, `torch`
* batches: `256`, `1024`, `4096`, `16384`
* reps: `3000` (warmup `200`), matches the A100 measurements.

Skipped combos:
* BlockRNN / BlockXfmr × TensorRT (no plans exist).
* batch 16384 × BlockXfmr (OOMs on L40 48 GB; auto-skipped after first OOM).

Total sweep time: ≈ **3–5 min** per SKU. Logs to stdout; failures are
non-fatal (sweep continues, JSONs that didn't get produced are flagged
in the run summary at the end).

### 5. Verify

```bash
python cloud/h100_l40_wcet/verify.py \
    --results_dir experiments/eE_lap_wcet/results \
    --expected   cloud/h100_l40_wcet/expected.json
```

This prints:
```
[verify] H100_PCIe-80GB:
  TinyMLP TRT b=4096    P99.9 =    25.4 µs   (budget 200 µs)  ✓
  TinyMLP ONNX b=4096   P99.9 =   174.2 µs   (budget 200 µs)  ✓
  TinyMLP torch b=4096  P99.9 =    87.1 µs   (budget 200 µs)  ✓
  ...
```

A red ✗ next to any P99.9 over 200 µs flags the architecture as
**disqualified** for that SKU — that is itself an eE-table-relevant
finding, not a failure of the kit.

### 6. Copy results back to your laptop

```bash
# From your laptop:
scp <node>:/workspace/SEER/experiments/eE_lap_wcet/results/lap_wcet_*<GPU_NAME>*.json \
    experiments/eE_lap_wcet/results/

# Then locally:
bash cloud/h100_l40_wcet/integrate.sh
```

`integrate.sh` runs `experiments/eE_lap_wcet/analyze.py` to refresh
`paper/figures/eE_wcet.pdf` with the new SKU bars and re-renders
`paper/sections/06_experiments.tex`'s table eE row block. Then it
prints a paste-ready snippet for the abstract and §1.

### 7. Iteration tip

Run **H100 first**, validate that the JSONs are well-formed, then
spin down the H100 and bring up an L40. Don't run both in parallel —
TensorRT plan rebuilds compete for the host's CPU and you waste money.

## Expected outcomes

Hypothesis derived from FLOPS scaling against A100's 33.8 µs P99.9:

| SKU | Predicted TinyMLP TRT b=4096 P99.9 | Source |
| --- | --- | --- |
| A100 SXM4-80GB | **33.8 µs** (measured) | repo |
| H100 SXM5-80GB | ≈ 18–25 µs | 2× SM throughput, same memory width |
| H100 PCIe-80GB | ≈ 22–30 µs | 1.6× throughput vs A100 |
| L40 (Ada) | ≈ 40–55 µs | 1.4× clock, 2× lower SM count |
| L40S | ≈ 35–48 µs | 12 % faster than L40 |

The headline claim survives if **all four SKUs** clear the 200 µs
budget at TinyMLP TRT b=4096. If L40 misses, that is itself paper-worthy
(suggests TinyMLP is not deployable on Ada; one of the larger archs
with quantisation might be needed).

## What this kit deliberately does NOT do

* No multi-tenant contention test on cloud — the §6.7 cross-card
  contention story is local-cluster (and is already in the paper).
* No NVLink probe — that requires a multi-GPU node (the existing
  A100×2 result is sufficient as one data point).
* No re-training of LAP on H100/L40. We deploy the same A100-trained
  TinyMLP plan; WCET is a deployment-side property.

## After integration: paper updates

Once the H100 / L40 JSONs are merged, three places in the paper need
updates. `integrate.sh` prints them as a checklist; the
specific snippets it generates are also written to
`cloud/h100_l40_wcet/paper_patches.txt` for cut-and-paste:

1. Abstract — add "(33.8 / 25 / 50 µs P99.9 on A100 / H100 / L40)" or
   the actual numbers.
2. §1 introduction headline bullet — change "On A100 SXM4-80GB"
   to "across A100 SXM4, H100 SXM5, and L40".
3. §6.7 (eE) Table II — add the H100 / L40 rows.
4. §8 discussion — strike the "H100 / L40 cross-hardware (out of scope)"
   paragraph.

If a SKU misses the budget at any (arch, backend, batch), the
checklist additionally instructs you to add an "architecture
disqualified on SKU X" note to §6.7 — turning a negative result into
a paper-worthy finding.
