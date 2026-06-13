# eF: Mixed-SLO workload scheduling

## Purpose

Same GPU, two workloads with different SLOs (chat-50ms TPOT + doc-1s
TTFT). Test whether SEER's PI controller can keep both miss ratios
below their respective targets, and whether admission control derived
from Lemma 2's *budget-from-SLO* inversion picks reasonable HBM
allocations.

Status: Active per Phase 6 (P6.13–P6.14). The multi-tenant
integration with vLLM is the hard part of this experiment; the
in-process driver here exercises the same policy code paths
end-to-end and produces the §6.8 figure.

## Run

```bash
bash run.sh
python analyze.py results
```
