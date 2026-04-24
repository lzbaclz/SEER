# E6: Policy Ablation

Four variants of SEER to separate the contributions of:

  1. **learned scoring** (LAP) vs. heuristic scoring
  2. **eviction-only** vs. **prefetch-only** vs. **joint** eviction+prefetch
  3. **IO-cost-aware** vs. quality-only utility

Matrix:

| variant             | scoring   | evict | prefetch | λ_io  |
|---------------------|-----------|-------|----------|-------|
| evict_heuristic     | H2O       | ✓     |          | 0     |
| evict_lap           | LAP       | ✓     |          | 0     |
| prefetch_lap        | LAP       |       | ✓        | 0     |
| joint_lap           | LAP       | ✓     | ✓        | 0     |
| joint_lap_io_aware  | LAP       | ✓     | ✓        | > 0   |

Run via the top-level runner with appropriate `--policy` + kwargs.
