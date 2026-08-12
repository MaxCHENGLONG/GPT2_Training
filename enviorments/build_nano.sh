#!/bin/bash
#SBATCH -A naiss2025-22-1730
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J build_nanogpt
#SBATCH -o /nobackup/proj/disk/naiss2025-22-1730/personal/licheng/logs/%x-%j.out

set -euo pipefail

BASE=/nobackup/proj/disk/naiss2025-22-1730/personal/licheng/GPT2_Training/enviorments
DEF=$BASE/nanogpt.def
SIF=$BASE/nanogpt.sif

export APPTAINER_CACHEDIR=$BASE/apptainer_cache
export APPTAINER_TMPDIR=$BASE/apptainer_tmp

apptainer build --fakeroot "$SIF" "$DEF"

apptainer exec --nv "$SIF" python -c \
  "import platform, torch; print(platform.machine(), torch.__version__, torch.cuda.is_available())"