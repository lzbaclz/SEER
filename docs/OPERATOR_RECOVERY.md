# Operator recovery: clearing the wedged CUDA / MPS state

R36e diagnostic. The seer conda env's `torch.cuda.is_available()`
hangs indefinitely because the NVIDIA MPS daemon is stuck. This
must be cleared **before** any GPU work in this repo (baseline
parity, multi-tenant, MIG A7, hot-path) can proceed.

## Symptom

```text
$ python -c "import torch; print(torch.cuda.is_available())"
# never returns; the process must be SIGKILL'd
```

Bisect of `seer.eval.runner` imports confirms the hang is exactly
at `torch.cuda.is_available()`. Background:

* `nvidia-cuda-mps-control -d` was started at some point (PID 911692)
* `nvidia-cuda-mps-server` is its child (PID 2889104, varies)
* Both persist across user logout; they keep an exclusive CUDA
  context that other clients block on

## Recovery (~30 sec, requires sudo)

```bash
# 1. Stop the MPS daemon cleanly first
echo quit | sudo nvidia-cuda-mps-control

# 2. If the daemon refused (most common; the control socket can be
#    stale), force-kill both processes
sudo pkill -9 -f nvidia-cuda-mps

# 3. Verify both GPUs are reachable without MPS
nvidia-smi --query-gpu=index,name --format=csv

# 4. Sanity-check torch
source /home/lzq/miniconda3/etc/profile.d/conda.sh && conda activate seer
python -c "import torch; print('cuda=', torch.cuda.is_available(), torch.cuda.device_count())"
# Should print: cuda= True 2
```

## After recovery — resume operator chain

Pick up `docs/mig_a7_runbook.md` → `experiments/eA_tail_latency/run_baseline_parity.sh`
→ `experiments/eF_mixed_slo/run_multitenant_seedsweep.sh` →
`python scripts/gen_claim_evidence_table.py && make paper && commit`.

The R36e commit (HEAD a6536b7) ships all scaffolding code + 11
hot-path tests + manifest entries 35--36 + claim-evidence rows.
A single fresh shell with the MPS daemon cleared executes the
chain in ~50 min.

## Why this happened in this session

Earlier in the session I launched several debug runs against
`--policy h2o --hbm_budget 0.05` that the runner could not finish
(model-load hang at the tight budget). Killing those leaked
processes is documented in `docs/mig_a7_runbook.md`. However the
deeper root cause is that the MPS daemon itself wedged — likely
when an earlier sweep died mid-CUDA-context-acquire and left the
daemon in a half-open state.

The sandbox correctly refused to let me kill `nvidia-cuda-mps-*`
PIDs (they belong to root, and pkill across user contexts on a
shared GPU host is a high-severity action), so this final step
must be done by the operator.
