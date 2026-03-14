# Pupper RL — Local Training Pipeline

Train a reinforcement learning locomotion policy for the Pupper V3 robot using your local GPU.

---

## Hardware & Software

| Item | Requirement |
|---|---|
| GPU | NVIDIA (RTX 5090 confirmed working) |
| CUDA | 12.x |
| Python | 3.10+ |
| OS | Ubuntu (with NVIDIA drivers installed) |

Installed package versions (as of setup):
- JAX 0.6.2 + CUDA 12
- MuJoCo 3.2.7 / MuJoCo MJX 3.2.7
- Brax 0.12.1
- Flax 0.10.2

---

## Files in This Folder

| File | Purpose |
|---|---|
| `Pupper_RL_LOCAL.ipynb` | **Main training notebook** — use this, not the PUBLIC version |
| `Pupper_RL_PUBLIC.ipynb` | Original Google Colab notebook (reference only) |
| `setup_local.sh` | One-time dependency installer (no sudo required) |
| `run_local.sh` | Headless runner — saves outputs to log file for debugging |
| `convert_to_local.py` | Script that generated the LOCAL notebook from the PUBLIC one |
| `pupperv3_mjx/` | Cloned robot MJX simulation library |
| `pupper_v3_description/` | Cloned robot URDF/MuJoCo model files |
| `wandb/` | Local W&B run cache |
| `output_*/` | Training output folders (checkpoints, videos, policy JSON) |

---

## One-Time Setup

Run once after a fresh clone or if packages change:

```bash
cd "/home/ttund/Tund/Pupper/Local pipeline"
./setup_local.sh
```

No sudo required — all system dependencies are already present on this machine.

**W&B is already authenticated.** The API key is saved in `~/.netrc` and will be picked up automatically.

---

## Running Training (Interactive)

```bash
cd "/home/ttund/Tund/Pupper/Local pipeline"
jupyter notebook Pupper_RL_LOCAL.ipynb
```

Then in the browser, run cells top to bottom (or **Run All**).

Cell outputs are saved into the `.ipynb` file automatically as cells execute.
Hit **Ctrl+S** before asking for help so Claude Code can read the latest outputs.

### Running Headlessly (unattended / overnight)

```bash
cd "/home/ttund/Tund/Pupper/Local pipeline"
./run_local.sh
```

Outputs are saved to:
- `run_output.log` — full stdout/stderr stream
- `Pupper_RL_EXECUTED.ipynb` — notebook with all cell outputs embedded

---

## Key Configuration

All training parameters are in cells under the **Config** section of the notebook.
Edit these before running. Common ones:

| Parameter | Location | Default | Notes |
|---|---|---|---|
| `num_timesteps` | `training_config.ppo` | 300M | Set to 1B for a better policy |
| `learning_rate` | `training_config.ppo` | 3e-4 | Use 1e-5 if training >300M steps |
| `num_envs` | `training_config.ppo` | 8192 | Parallel environments |
| `n_obstacles` | `training_config` | 0 | Set 5–20 to train over obstacles |
| `episode_length` | `training_config.ppo` | 500 | Steps per episode |
| `hidden_layer_sizes` | `policy_config` | (256,128,128,128) | Policy network architecture |
| `activation` | `policy_config` | `elu` | Must be supported by RTNeural for deployment |

After changing any config value, re-run all cells from the top so values propagate correctly.

---

## Training Outputs

Each run creates a folder: `output_<wandb-run-name>/`

| File | Contents |
|---|---|
| `policy_<run-name>_max_reward_<N>.json` | **Deployable policy** for the robot's neural_controller |
| `mjx_params_<datetime>` | Raw JAX model weights |
| `*.mp4` / `*.html` | Policy visualization videos generated during training |

The policy JSON and model weights are also uploaded to Weights & Biases automatically.

---

## Deploying to the Robot

Download `policy_*.json` from the `output_*/` folder (or from W&B).

See the full deployment guide:
https://pupper-v3-documentation.readthedocs.io/en/latest/development/modifying_code.html

---

## Debugging with Claude Code

1. Run the notebook interactively as normal
2. When a cell errors, hit **Ctrl+S** to save
3. Tell Claude Code to check the notebook — it will read `Pupper_RL_LOCAL.ipynb` and see all cell outputs and tracebacks

For headless runs, tell Claude Code to check `run_output.log`.

### Known Issues & Fixes

**`jax.tree_map` AttributeError**
Affects `pupperv3_mjx/domain_randomization.py` line 93 when using JAX 0.6+.
Fixed by patching `~/.local/lib/python3.10/site-packages/pupperv3_mjx/domain_randomization.py`:
```python
# Old (broken on JAX 0.6+):
in_axes = jax.tree_map(lambda x: None, sys)
# Fixed:
in_axes = jax.tree.map(lambda x: None, sys)
```
This patch has already been applied.

---

## Monitoring Training

Runs are logged to Weights & Biases project **pupperv3-mjx-rl**.

W&B account: `nathankau` / org: `lastfeetrobotics` (configured in `~/.netrc`)

Visit https://wandb.ai to view live training curves, reward plots, and policy videos.

---

## Useful Links

- Pupper V3 docs: https://pupper-v3-documentation.readthedocs.io
- pupperv3-mjx repo: https://github.com/Nate711/pupperv3-mjx
- MuJoCo Playground (for custom tasks): https://github.com/google-deepmind/mujoco_playground
- Community Discord: https://discord.com/invite/qbmaU8NmP2
