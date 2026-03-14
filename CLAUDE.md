# Claude Code Context — Pupper Local Pipeline

## Repository
- GitHub: https://github.com/TundTT/pupper-local-pipeline.git
- Local path: `/home/ttund/Tund/Pupper/Local pipeline`
- Git credentials stored in `~/.git-credentials` — Claude can push/pull freely

## User
- Name: Tund | Email: Tundt23@gmail.com | GitHub: TundTT

## What This Project Is
Local version of the Pupper V3 RL training pipeline, converted from Google Colab to run on local GPU hardware (NVIDIA RTX 5090, CUDA 12.8, Python 3.10).

## Key Files
| File | Purpose |
|---|---|
| `Pupper_RL_LOCAL.ipynb` | Main training notebook — run this |
| `setup_local.sh` | One-time dependency installer (no sudo needed) |
| `run_local.sh` | Headless runner, saves outputs to `run_output.log` |
| `pupperv3_mjx/` | Patched local copy of simulation library (detached from upstream) |
| `pupper_v3_description/` | Robot model files (detached from upstream) |
| `README.md` | Full usage documentation |

## Running Training
```bash
jupyter notebook Pupper_RL_LOCAL.ipynb
# Then Kernel → Restart & Run All
```

## Known Patches Applied
- `pupperv3_mjx/pupperv3_mjx/domain_randomization.py` line 93:
  `jax.tree_map` → `jax.tree.map` (removed in JAX 0.6.0)
- Both `pupperv3_mjx/` and `pupper_v3_description/` are detached from their
  upstream git remotes — edit files directly, re-run cell 16 to reinstall

## Weights & Biases
- Authenticated via `~/.netrc` (no env var needed)
- Project: `pupperv3-mjx-rl`

## Debugging
- Save notebook (Ctrl+S) then ask Claude to check `Pupper_RL_LOCAL.ipynb`
- For headless runs, check `run_output.log`

## Notebook Edit Workflow
- Claude edits `.py` source files directly when possible
- For notebook cell changes, Claude will paste corrected code as text
- If Claude edits the `.ipynb` directly, it will say "hit Revert in Jupyter"

## Dependencies (already installed)
- JAX 0.6.2 + CUDA 12
- MuJoCo 3.2.7 / MuJoCo MJX 3.2.7
- Brax 0.12.1, Flax 0.10.2
- ffmpeg, NVIDIA EGL ICD — already present system-wide (no sudo needed)
