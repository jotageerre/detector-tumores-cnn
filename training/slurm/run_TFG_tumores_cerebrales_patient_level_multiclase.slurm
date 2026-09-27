#!/bin/bash
# ==============================================================================
# run_TFG_tumores_cerebrales_patient_level_multiclase.slurm
# ==============================================================================
# Ejecuta TFG_tumores_cerebrales_patient_level_multiclase.ipynb de principio a
# fin en el cluster HPC de la Universidad de Sevilla, vía `jupyter nbconvert
# --execute`, y conserva notebook fuente + notebook ejecutado + logs + resultados.
#
# TODOS los valores marcados con <...> son PLACEHOLDERS: rellénalos con los
# datos reales de tu cuenta/partición antes de lanzar el job. No se ha inventado
# ningún dato de cluster que no se conociera ya (partición, cuenta, tiempo,
# memoria y tipo de GPU concretos dependen de tu asignación real).
# ==============================================================================
#SBATCH --job-name=tfg_tumores_multiclase
#SBATCH --partition=<PARTITION>              # p.ej. la partición GPU de tu cuenta
#SBATCH --account=<ACCOUNT>                  # tu cuenta/proyecto en el cluster
#SBATCH --gres=gpu:<GPU>:1                   # p.ej. gpu:a30:1 -- usa el tipo de GPU que tengas asignado
#SBATCH --cpus-per-task=8
#SBATCH --mem=<MEMORY>                       # p.ej. 32G -- ajusta a lo disponible/necesario
#SBATCH --time=<TIME>                        # p.ej. 04:00:00 -- el run 70436 (binario+control) tardó 28 min;
                                              # esta versión entrena 5 modelos multiclase adicionales, deja margen
#
# NOTA sobre --output/--error: Slurm resuelve y ABRE estos ficheros al arrancar
# el job, ANTES de ejecutar una sola línea del script -- si aquí se pusiera una
# ruta con subcarpeta (p.ej. "logs/%x_%j.out") y esa subcarpeta no existiera ya
# en el directorio de envío en el momento de hacer `sbatch`, el job fallaría de
# inmediato con un error de Slurm, no del notebook. Para no depender de que el
# usuario recuerde crear "logs/" antes de cada sbatch, se escriben aquí sin
# subcarpeta (siempre existe el directorio desde el que se invoca `sbatch`) y
# el propio script los traslada a "$TFG_PROJECT_ROOT/logs/" al terminar (bloque
# ENTORNO más abajo), tanto si el job acaba bien como si falla.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Directorio desde el que se ha invocado `sbatch` -- es donde Slurm ha creado
# los ficheros %x_%j.out/.err de arriba. Se captura ANTES de hacer `cd`, para
# poder localizarlos y moverlos al final pase lo que pase.
_DIR_ENVIO="${SLURM_SUBMIT_DIR:-$PWD}"
_LOG_OUT="$_DIR_ENVIO/${SLURM_JOB_NAME:-tfg_tumores_multiclase}_${SLURM_JOB_ID:-sinid}.out"
_LOG_ERR="$_DIR_ENVIO/${SLURM_JOB_NAME:-tfg_tumores_multiclase}_${SLURM_JOB_ID:-sinid}.err"

# ------------------------------------------------------------------ RUTAS ---
# PROJECT_ROOT es la única variable de entorno que el notebook necesita para
# situar todas sus rutas (ver Sección 0 del notebook). Ajusta esta ruta a
# donde tengas el proyecto en el cluster.
export TFG_PROJECT_ROOT="${TFG_PROJECT_ROOT:-$HOME/TFG_tumores/tfg_run}"
mkdir -p "$TFG_PROJECT_ROOT"/{data,results,frozen,logs}

# Al terminar (éxito, fallo o cancelación) se copian los logs de Slurm a
# $TFG_PROJECT_ROOT/logs/, que es donde vive el resto de artefactos de esta
# ejecución. Se copian (no se mueven) porque Slurm puede seguir escribiendo en
# el fichero original hasta que el proceso termina del todo.
_copiar_logs() {
    cp -f "$_LOG_OUT" "$TFG_PROJECT_ROOT/logs/" 2>/dev/null || true
    cp -f "$_LOG_ERR" "$TFG_PROJECT_ROOT/logs/" 2>/dev/null || true
}
trap _copiar_logs EXIT

# NOTEBOOK_ENTRADA y NOTEBOOK_SALIDA son rutas relativas a $TFG_PROJECT_ROOT:
# hace falta situarse ahí antes de usarlas.
cd "$TFG_PROJECT_ROOT"

NOTEBOOK_ENTRADA="notebooks/TFG_tumores_cerebrales_patient_level_multiclase.ipynb"
NOTEBOOK_SALIDA="results/TFG_tumores_cerebrales_patient_level_multiclase_EXECUTED.ipynb"

# --------------------------------------------------------- SPLIT CONGELADO --
# El split multiclase congelado (ver baseline_multiclase_cheng.md) debe existir
# en $TFG_PROJECT_ROOT/frozen/ ANTES de lanzar este job. El propio notebook se
# detiene con un error si no lo encuentra o si no coincide con el manifiesto
# actual -- este script no lo genera ni lo copia por ti.
if [ ! -f "$TFG_PROJECT_ROOT/frozen/split_multiclase_cheng.json" ]; then
    echo "[AVISO] No se encuentra $TFG_PROJECT_ROOT/frozen/split_multiclase_cheng.json"
    echo "        Cópialo ahí antes de relanzar; el notebook se detendrá igualmente"
    echo "        si falta, pero es mejor saberlo ahora que a mitad de ejecución."
fi

# --------------------------------------------------------------- ENTORNO ---
module purge
module load "<MODULO_PYTHON>"                # p.ej. module load Python/3.10.8 (usa el nombre real de tu módulo, SIN comillas ni "<>")
module load "<MODULO_CUDA>"                  # p.ej. module load CUDA/12.2 (usa el nombre real de tu módulo, SIN comillas ni "<>")
# Si el proyecto usa un entorno virtual/conda ya preparado, actívalo aquí:
# source <RUTA_A_TU_VENV>/bin/activate

echo "===== INFORMACIÓN DEL ENTORNO ====="
echo "Fecha           : $(date)"
echo "Nodo            : $(hostname)"
echo "PROJECT_ROOT    : $TFG_PROJECT_ROOT"
python3 --version
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi no disponible"
nvcc --version 2>/dev/null || echo "nvcc no disponible (CUDA vía TensorFlow/cuDNN empaquetado)"
echo "===================================="

# ------------------------------------------------- MODO DE EJECUCIÓN (RUN_MODE) --
# RUN_MODE y RUN_BINARY_SECONDARY se fijan DENTRO del notebook (Sección 1.5),
# no aquí: este script no los sobrescribe. Antes de lanzar, confirma en el
# propio notebook qué fase quieres ejecutar esta vez (baseline_only /
# multiclass_benchmark / full) -- ver baseline_multiclase_cheng.md y la
# explicación de la Sección 1.5.
echo "Recuerda: RUN_MODE / RUN_BINARY_SECONDARY se fijan dentro del notebook (Sección 1.5)."

# --------------------------------------------------------- EJECUCIÓN -------
# jupyter nbconvert --execute corre el notebook de principio a fin como si se
# ejecutara "Run All" y escribe una copia con las salidas (celdas, prints,
# figuras) en NOTEBOOK_SALIDA, sin tocar el notebook fuente.
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output "$(basename "$NOTEBOOK_SALIDA")" \
    --output-dir "$(dirname "$NOTEBOOK_SALIDA")" \
    "$NOTEBOOK_ENTRADA"

echo "===== EJECUCIÓN TERMINADA ====="
echo "Notebook ejecutado : $NOTEBOOK_SALIDA"
echo "Resultados en      : $TFG_PROJECT_ROOT/results/"
echo "  binary/          : checkpoints, métricas y predicciones de la tarea binaria"
echo "  tumor_type/      : checkpoints, métricas y predicciones de la tarea multiclase"
echo "Logs de Slurm en   : $TFG_PROJECT_ROOT/logs/ (copiados automáticamente al salir)"
echo "================================"
