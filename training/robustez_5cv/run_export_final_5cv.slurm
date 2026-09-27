#!/bin/bash -l

#SBATCH --job-name=tfg_export_5cv
#SBATCH --partition=main
#SBATCH --account=i3us
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/shared/home/FYK3492/TFG_tumores/exports/tfg_export_5cv_%j.out
#SBATCH --error=/shared/home/FYK3492/TFG_tumores/exports/tfg_export_5cv_%j.err

set -eo pipefail

ROOT=/shared/home/FYK3492/TFG_tumores
EXPORTS=$ROOT/exports
ARCHIVE=$EXPORTS/TFG_tumores_FINAL_5CV_20260904.tar.gz
SHA=$ARCHIVE.sha256

mkdir -p "$EXPORTS"

echo "============================================================"
echo "EXPORTACIÓN FINAL TFG + VALIDACIÓN 5CV"
echo "Fecha: $(date)"
echo "Nodo : $(hostname)"
echo "Job  : ${SLURM_JOB_ID:-N/A}"
echo "============================================================"

cd "$ROOT"

rm -f "$ARCHIVE" "$SHA"

echo
echo "Tamaño del proyecto:"
du -sh tfg_run

echo
echo "Creando:"
echo "$ARCHIVE"

tar -czf "$ARCHIVE" tfg_run

echo
echo "Calculando SHA-256..."
sha256sum "$ARCHIVE" | tee "$SHA"

echo
echo "Comprobando integridad del tar.gz..."
tar -tzf "$ARCHIVE" > /dev/null

echo
echo "Archivo creado:"
ls -lh "$ARCHIVE" "$SHA"

echo
echo "============================================================"
echo "EXPORTACIÓN COMPLETADA CORRECTAMENTE"
echo "============================================================"
