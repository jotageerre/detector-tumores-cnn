#!/bin/bash -l

#SBATCH --job-name=tfg_5cv_resnet
#SBATCH --partition=main
#SBATCH --account=i3us
#SBATCH --gres=gpu:a30:1
#SBATCH --nodelist=gpu02
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-4%1
#SBATCH --output=tfg_5cv_resnet_%A_%a.out
#SBATCH --error=tfg_5cv_resnet_%A_%a.err

set -eo pipefail

source /etc/profile

module purge
module load Python/3.10.8-GCCcore-12.2.0

source /shared/home/FYK3492/venvs/tfg/bin/activate

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" -type d -name lib -print 2>/dev/null | paste -sd: -)

if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
fi

LIBDEVICE=$(find "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia" -name "libdevice.10.bc" -print -quit 2>/dev/null || true)

if [ -n "$LIBDEVICE" ]; then
    CUDA_PIP_ROOT=$(dirname "$(dirname "$(dirname "$LIBDEVICE")")")
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_PIP_ROOT"
fi

cd /shared/home/FYK3492/TFG_tumores/tfg_run

echo "============================================================"
echo "TFG - RESNET50 ROBUSTEZ 5CV"
echo "Array job : ${SLURM_ARRAY_JOB_ID}"
echo "Task      : ${SLURM_ARRAY_TASK_ID}"
echo "Fold      : $((SLURM_ARRAY_TASK_ID + 1))"
echo "Nodo      : $(hostname)"
echo "Fecha     : $(date)"
echo "============================================================"

nvidia-smi

python ejecutar_fold_5cv_robustez.py --fold-index "${SLURM_ARRAY_TASK_ID}"
