# KFN Canonical Reruns — Internal Check 3

Minimal canonical package. **One shared core** (`kfn_core.py`) + **thin
runners** (one per manuscript artifact). Table VI (controlled capacity reuse)
needs **no rerun** — the reviewer verified it as executed with matching
aggregates; we reuse `New/Controlled_Capacity_Reuse/` as-is.

## Files (6 total)

| File | Purpose | Manuscript artifact |
|------|---------|---------------------|
| `kfn_core.py` | THE canonical implementation (models, SVD novelty, replay, baselines, metrics) | — |
| `run_tables1_3_7.py` | Two-task studies (CIFAR-10 / CIFAR-100 CIL / EuroSAT) | Tables I, III, VII |
| `run_table2.py` | 10-task Split CIFAR-100 | Table II + novelty trajectory |
| `run_table4.py` | Isolation ablation (4 variants) | Table IV |
| `run_efficiency.py` | Compute/memory + 10-task parameter curve | Table VIII + Figure 7 |
| `smoke_test.py` | Fast end-to-end validation (~15 min) | — |

## Canonical protocol (in `kfn_core.py`, used by ALL runners)

- **No test leakage**: train/val/reference splits from *training* data; novelty
  on reference only; test touched exactly once per task.
- **SVD novelty** = manuscript Eqs. (5)–(9): pool → L2 norm (ε_norm=1e-8) →
  basis (n_ref=256) → P = U_r U_rᵀ (ε_svd=1e-6) → ‖(I−P)f̂‖₂. Realised rank recorded.
- **τ = 0.2**, d_high=64, d_low=0.
- **m = 20/class** replay with byte accounting per task.
- **~2.8M specialist** (3→64→128→256→d=64); per-component parameter breakdown.
- **Recursion preserved**: all prior projections carried over; classifier width
  = 64 + Σd_new; α = max(0.1, 1−ν) schedule + learned σ(g).
- **Raw per-seed JSON** for every seed (no omission); aggregate.json reports
  mean±std plus retention as ratio-of-means AND mean-of-ratios.

## Environment

```bash
pip install -r requirements.txt   # torch>=2.1, torchvision>=0.16, numpy>=1.24
python -c "import torch; print(torch.cuda.is_available())"
```

Any CUDA GPU with ≥8 GB VRAM. CPU works but is slow.

## Cloud execution plan

```bash
# 0) validate everything end-to-end first (~15 min)
python smoke_test.py

# 1) two-task studies — independent, can run in parallel on separate GPUs
python run_tables1_3_7.py --table 1 --seeds 0 1 2 3 4   # Table I   (~2-5 h)
python run_tables1_3_7.py --table 3 --seeds 0 1 2 3 4   # Table III (~3-7 h)
python run_tables1_3_7.py --table 7 --seeds 0 1 2 3 4   # Table VII (~2-5 h)

# 2) ablation (~3-6 h)
python run_table4.py --seeds 0 1 2 3 4

# 3) compute/memory table (~1-2 h)
python run_efficiency.py --seeds 0 1 2 3 4 --skip-10task

# 4) LONG JOBS — run last, in parallel with each other if 2 GPUs
python run_table2.py --seeds 0 1 2 3 4                   # Table II  (~10-24 h)
python run_efficiency.py --seeds 0 1 2 3 4               # Figure 7  (~10-24 h)
```

Jobs are **resumable**: each (method, seed) result is written immediately and
skipped on re-run, so an interrupted job resumes by re-running the same command.

## What comes back

Collect and return:

```bash
tar czf kfn_results.tar.gz results_table* results_efficiency smoke_*/
nvidia-smi > gpu.txt
pip freeze > env.txt
```

Plus any stderr/stdout logs if something failed. Every number needed to update
Tables I–IV, VII, VIII, Figure 7, and the novelty trajectory lives in the
`aggregate.json` files.
