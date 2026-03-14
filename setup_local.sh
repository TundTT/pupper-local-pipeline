#!/bin/bash
# Setup script to install dependencies for running Pupper RL pipeline locally.
# Run once before launching the notebook.
# Requires: NVIDIA GPU with CUDA 12.x, Python 3.10+
# No sudo required - all system dependencies are already present.

set -e

echo "=== Checking system dependencies ==="
command -v ffmpeg >/dev/null && echo "ffmpeg: OK" || echo "WARNING: ffmpeg not found - video rendering may fail"
[ -f "/usr/share/glvnd/egl_vendor.d/10_nvidia.json" ] && echo "NVIDIA EGL ICD: OK" || echo "WARNING: NVIDIA EGL ICD config not found"
[ -f "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0" ] && echo "libEGL_nvidia: OK" || echo "WARNING: libEGL_nvidia.so.0 not found"

echo ""
echo "=== Installing Python packages ==="

# JAX with CUDA 12 support - keeps current version (0.6.x) compatible with RTX 5090
pip install --upgrade "jax[cuda12]"

# Core simulation stack (versions matching original Colab notebook)
pip install mujoco==3.2.7 mujoco-mjx==3.2.7
pip install brax==0.12.1
pip install flax==0.10.2
pip install "orbax==0.1.9"

# Supporting libraries
pip install wandb
pip install mediapy
pip install plotly
pip install ml_collections
pip install etils[epath]
pip install "black[jupyter]"
pip install ipywidgets

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Before running the notebook, set your W&B API key:"
echo "  export WANDB_API_KEY=your_key_here"
echo ""
echo "Then launch Jupyter:"
echo "  cd \"/home/ttund/Tund/Pupper/Local pipeline\""
echo "  jupyter notebook Pupper_RL_LOCAL.ipynb"
echo ""
echo "Or with JupyterLab:"
echo "  jupyter lab"
