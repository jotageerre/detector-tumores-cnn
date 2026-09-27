#!/bin/bash -l

#SBATCH --job-name=tfg_5cv_split
#SBATCH --partition=main
#SBATCH --account=i3us
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=tfg_5cv_split_%j.out
#SBATCH --error=tfg_5cv_split_%j.err

set -eo pipefail

source /etc/profile

module purge
module load Python/3.10.8-GCCcore-12.2.0

source /shared/home/FYK3492/venvs/tfg/bin/activate

export PYTHONUNBUFFERED=1

cd /shared/home/FYK3492/TFG_tumores/tfg_run

echo "============================================================"
echo "TFG - PREPARACIÓN 5CV DE ROBUSTEZ"
echo "============================================================"
echo "Fecha : $(date)"
echo "Nodo  : $(hostname)"
echo "Job   : ${SLURM_JOB_ID:-N/A}"
echo

which python
python --version

echo
echo "Generando únicamente el particionado 5-fold..."
echo "NO se entrena ningún modelo en este job."
echo

python preparar_5cv_robustez.py

echo
echo "============================================================"
echo "PARTICIONADO 5CV FINALIZADO"
echo "============================================================"
