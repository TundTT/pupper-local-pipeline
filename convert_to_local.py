#!/usr/bin/env python3
"""
Converts Pupper_RL_PUBLIC.ipynb (Google Colab) to Pupper_RL_LOCAL.ipynb
for running locally with CUDA GPU hardware.

Run: python3 convert_to_local.py
"""

import json, copy, re

INPUT  = "Pupper_RL_PUBLIC.ipynb"
OUTPUT = "Pupper_RL_LOCAL.ipynb"

with open(INPUT) as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    """Return source as a single string."""
    return "".join(cell["source"])

def set_src(cell, text):
    """Replace cell source with lines list (keeps trailing newlines intact)."""
    lines = text.splitlines(keepends=True)
    cell["source"] = lines
    cell["outputs"] = []          # clear stale outputs
    cell["execution_count"] = None

# --------------------------------------------------------------------------
# Cell-by-cell patches
# --------------------------------------------------------------------------

for cell in cells:
    s = src(cell)

    # ── Cell: google.colab userdata → env-var lookup ──────────────────────
    if "from google.colab import userdata" in s:
        set_src(cell, """\
import os
# Set WANDB_API_KEY in your shell before launching Jupyter, e.g.:
#   export WANDB_API_KEY=your_key_here
wandb_key = os.environ.get('WANDB_API_KEY', None)
""")

    # ── Cell: MuJoCo installation check – remove google.colab import ──────
    elif "from google.colab import files" in s:
        s = s.replace("from google.colab import files\n", "")
        # nvidia-smi check still works locally; keep it
        set_src(cell, s)

    # ── Cell: JAX install – target CUDA 12 (no CPU-only downgrade) ────────
    elif "pip install jax==0.5.0 jaxlib==0.5.0" in s:
        set_src(cell, """\
# Install JAX with CUDA 12 support.
# We install the latest compatible version instead of pinning to 0.5.0
# so that it works on modern GPUs (RTX 5090, Blackwell architecture).
!pip install -q --upgrade "jax[cuda12]"
""")

    # ── Cell: repo git-hash helper – use subprocess instead of %cd ────────
    elif "get_hash" in s and "%cd /content/pupperv3_mjx" in s:
        set_src(cell, """\
import subprocess, os

def get_hash(path):
    return subprocess.check_output(
        ['git', '-C', path, 'rev-parse', 'HEAD']
    ).strip().decode('utf-8')

repo_config = config_dict.ConfigDict()
repo_config.pupperv3_mjx_hash = get_hash('pupperv3_mjx')
repo_config.pupper_v3_description_hash = get_hash('pupper_v3_description')
""")

    # ── Cell: gdrive export path → local path ─────────────────────────────
    elif "gdrive_save_dir" in s and "MyDrive" in s:
        s = s.replace(
            "'/content/drive/MyDrive/pupper_policies'",
            "'./pupper_policies'"
        )
        set_src(cell, s)

    # ── Cell: runtime.unassign() – drop the Colab shutdown call ───────────
    elif "runtime.unassign" in s:
        set_src(cell, """\
# (Colab runtime shutdown removed – not needed locally)
import time
time.sleep(10)
print("Training finished.")
""")

# --------------------------------------------------------------------------
# Notebook-level metadata: remove Colab-specific bits
# --------------------------------------------------------------------------
meta = nb.get("metadata", {})
meta.pop("colab", None)
meta.setdefault("kernelspec", {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
})
meta.setdefault("language_info", {"name": "python", "version": "3.10.12"})
nb["metadata"] = meta

# --------------------------------------------------------------------------
# Write output
# --------------------------------------------------------------------------
with open(OUTPUT, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {OUTPUT}")
print()
print("Key changes made:")
print("  • Replaced google.colab.userdata with os.environ WANDB_API_KEY lookup")
print("  • Removed google.colab.files import from MuJoCo check cell")
print("  • JAX install cell now uses jax[cuda12] (supports RTX 5090 / Blackwell)")
print("  • Git-hash cell no longer uses %cd /content/... magic")
print("  • Export path changed from Google Drive to ./pupper_policies")
print("  • runtime.unassign() removed")
