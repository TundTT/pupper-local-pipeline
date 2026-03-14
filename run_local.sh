#!/bin/bash
# Runs Pupper_RL_LOCAL.ipynb headlessly and saves outputs so they can be inspected.
# Outputs go to:
#   run_output.log          - full stdout/stderr stream
#   Pupper_RL_EXECUTED.ipynb - notebook with all cell outputs embedded

set -e

NOTEBOOK="Pupper_RL_LOCAL.ipynb"
EXECUTED="Pupper_RL_EXECUTED.ipynb"
LOG="run_output.log"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

echo "Starting at $(date)" | tee "$LOG"
echo "Notebook : $NOTEBOOK"  | tee -a "$LOG"
echo "Executed : $EXECUTED"  | tee -a "$LOG"
echo "Log      : $LOG"       | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Set MuJoCo to use GPU EGL rendering (no display needed)
export MUJOCO_GL=egl
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "$EXECUTED" \
    "$NOTEBOOK" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
echo "Cell outputs saved to: $EXECUTED"
echo "Full log saved to:     $LOG"
