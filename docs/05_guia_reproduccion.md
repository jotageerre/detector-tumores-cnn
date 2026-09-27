**Idioma:** **Español** · [English](en/05_reproduction_guide.md) · [Français](fr/05_guide_reproduction.md)

# 5. Guía de reproducción paso a paso

Esta guía explica cómo recrear los experimentos del TFG, de la ruta más rápida a la más fiel al original. Resume y amplía el Anexo G (instrucciones de reproducibilidad).

> **Qué significa "reproducir" aquí.** Se reproduce el **procedimiento**: mismos datos, mismo split, misma configuración, mismo protocolo. No se garantizan las mismas cifras al último decimal, porque el determinismo estricto de GPU no estaba activado (Anexo G.4). Espera resultados del mismo orden, no idénticos.

## 0. Requisitos

| | Mínimo | Usado en el TFG |
|---|---|---|
| Python | 3.10 | 3.10.8 |
| TensorFlow | 2.15.1 | 2.15.1 |
| GPU | cualquier NVIDIA con ≥ 8 GB (en CPU funciona, pero tarda horas) | NVIDIA A30 24 GB, CUDA 12.2 |
| Disco | ~3 GB (ZIPs + PNG) | |
| SO | Linux o **WSL2** en Windows (TensorFlow ya no usa GPU en Windows nativo) | Linux (clúster HPC) |

```bash
git clone https://github.com/<tu-usuario>/detector-tumores-cnn.git
cd detector-tumores-cnn/training
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # en Linux con GPU: pip install "tensorflow[and-cuda]==2.15.1"
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

> **scikit-learn < 1.8.** La versión 1.8 cambió cómo baraja `StratifiedGroupKFold` y genera folds distintos. Con las versiones 1.5.2, 1.6.1 y 1.7.2 se obtienen byte a byte los folds archivados (comprobado).

---

## Ruta A — Rápida: preparar los datos y entrenar el modelo final (≈ 30 min con GPU)

### A.1 Descargar Cheng y construir el manifiesto

```bash
python herramientas/construir_manifest_cheng.py \
    --root ./tfg_run \
    --split ../data/splits/split_multiclase_cheng.csv
```

Qué hace: descarga los 4 ZIP de figshare (~880 MB) verificando el MD5, convierte los 3.064 `.mat` a PNG con la función de preprocesado original, asigna a cada paciente su subconjunto del split congelado y **comprueba que la huella del split es `fda7e2da…`**. Si algo no cuadra, se detiene.

Salida esperada al final:

```
  huella del split : fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092
OK: .../tfg_run/results/artifacts/manifest_final.csv  (3064 cortes, 233 pacientes, split verificado)
tumor_type  glioma  meningioma  pituitario
subset
test            12          10          13
train           63          62          38
val             14          10          11
```

Si figshare bloquea la descarga automática (algunas redes devuelven 403), descarga los 4 ZIP a mano desde la [página del dataset](https://doi.org/10.6084/m9.figshare.1512427), descomprime los `.mat` en `tfg_run/data/raw/cheng_mat/` y ejecuta el script con `--sin-descarga`.

### A.2 Entrenar el modelo final y evaluarlo una vez en test

```bash
python entrenar_modelo_final.py --root ./tfg_run
```

- Entrena ResNet50‑FT con los 198 pacientes de train+val (6 épocas de cabeza + 6 de fine-tuning, semilla 42).
- Guarda `tfg_run/results/tumor_type/final/resnet50_ft_cheng_final.keras` y **`.h5`** (este es el que usa el backend de Cloud Run).
- Evalúa **una sola vez** los 35 pacientes de test y escribe `resultado_evaluacion_final.json`, las predicciones por corte y por paciente, y los ficheros `.lock`. Si vuelves a lanzarlo en la misma carpeta, se niega a abrir el test otra vez.

Referencia del TFG: balanced accuracy 93,89 %, accuracy 94,29 %, macro-F1 93,98 %.

> Este script es una **reimplementación** del protocolo C.1: el notebook exacto del job 70930 no se conserva. Reutiliza línea a línea la receta de `robustez_5cv/ejecutar_fold_5cv_robustez.py`, que sí es código original.

---

## Ruta B — Validación de robustez de 5 folds con los scripts originales

Los scripts de `training/robustez_5cv/` son **exactamente** los que se ejecutaron en el clúster (jobs 71101, 71102, 71107). Tienen la ruta del clúster escrita a mano; cámbiala por la tuya (el hash del script cambiará, es normal):

```bash
cd training/robustez_5cv
RUTA=$(realpath ../tfg_run)          # la carpeta --root de la Ruta A
sed -i "s#/shared/home/FYK3492/TFG_tumores/tfg_run#$RUTA#" preparar_5cv_robustez.py ejecutar_fold_5cv_robustez.py agregar_5cv_robustez.py
# carpetas que ya existían en el clúster y los scripts dan por hechas
mkdir -p "$RUTA/frozen/robustez_5cv" "$RUTA/results/diagnostics/robustez_5cv"
# el protocolo escrito antes de ejecutar (los scripts guardan su SHA-256 en cada resultado)
cp ../../docs/protocolos/C2_validacion_robustez_5cv.md "$RUTA/frozen/robustez_5cv/protocolo_5cv_robustez.md"

python preparar_5cv_robustez.py                    # 1) crea los 5 folds (StratifiedGroupKFold, seed 20260904)
for i in 0 1 2 3 4; do                             # 2) un modelo independiente por fold (seeds 1001..1005)
    python ejecutar_fold_5cv_robustez.py --fold-index $i
done
python agregar_5cv_robustez.py                     # 3) verifica cobertura de los 233 pacientes y agrega
```

Comprobación de que tus folds son los del TFG:

```bash
diff <(python -m json.tool $RUTA/results/tumor_type/robustez_5cv/splits/fold_1.json) \
     <(python -m json.tool ../../data/splits/robustez_5cv/fold_1.json) && echo "fold 1 idéntico"
```

Compara tu `resultado_5cv_validacion_robustez.json` con [`results/robustez_5cv/`](../results/robustez_5cv): media de referencia 89,44 % ± 4,64 pp.

---

## Ruta C — Pipeline completo original (notebook)

`training/notebooks/02_pipeline_patient_level_multiclase.ipynb` es el pipeline completo tal como se usó en el benchmark multiclase: descarga Cheng **e IXI**, construye el manifiesto de 5.872 imágenes, detecta duplicados, verifica el split congelado, reproduce el baseline histórico y ejecuta el benchmark de arquitecturas.

1. Define la carpeta de trabajo: `export TFG_PROJECT_ROOT=$PWD/tfg_run_nb` (en Colab usa `/content/tfg_tumores` por defecto).
2. Copia el split congelado: `mkdir -p $TFG_PROJECT_ROOT/frozen && cp ../data/splits/split_multiclase_cheng.* $TFG_PROJECT_ROOT/frozen/`
3. Elige qué ejecutar en la **Sección 1.5** del notebook:
   - `RUN_MODE = "baseline_only"` → datos + split + baseline CNN propia (control de regresión).
   - `RUN_MODE = "multiclass_benchmark"` → además, las 5 arquitecturas con GroupCV de 3 folds (Tabla 4.2).
   - `RUN_MODE = "full"` y `RUN_BINARY_SECONDARY = True` → además, la tarea binaria histórica con IXI.
4. Ejecuta todo: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/02_pipeline_patient_level_multiclase.ipynb`

Avisos:
- El notebook se **detiene** si el split recalculado no coincide con el congelado: es intencionado.
- La descarga de IXI puede fallar (el servidor de Imperial bloquea algunas IPs); el notebook reintenta y usa Zenodo como alternativa.
- El **control de regresión** del baseline puede detener la ejecución (le pasó al job 70886; ver [04_resultados.md](04_resultados.md#41-baseline-histórico-cnn-propia-desde-cero)).
- Las fases de fine-tuning y fusión (protocolos C.3 y C.4) **no están en esta versión del notebook** (Anexo A.3). Sus protocolos están en [`protocolos/`](protocolos) para quien quiera reimplementarlos.

`01_reconstruccion_dataset_y_control_3clases.ipynb` es la versión anterior de la etapa de reconstrucción (job 70436), con salidas; se conserva como referencia histórica. Consulta [`training/notebooks/baseline_multiclase_cheng.md`](../training/notebooks/baseline_multiclase_cheng.md) para la relación entre versiones.

---

## Ruta D — En un clúster con SLURM

Los ficheros `.slurm` son los originales. Adáptalos a tu clúster (partición, cuenta, módulos y rutas):

| Script | Qué lanza | Recursos originales |
|---|---|---|
| `training/slurm/run_TFG_tumores_cerebrales_patient_level_multiclase.slurm` | notebook 02 vía `nbconvert` | plantilla con `<PARTITION>`, `<ACCOUNT>`… |
| `training/robustez_5cv/run_preparar_5cv_robustez.slurm` | creación de folds | 2 CPU, 4 GB, sin GPU |
| `training/robustez_5cv/run_5cv_robustez_array.slurm` | los 5 folds como *job array* (`--array=0-4%1`) | 1× A30, 8 CPU, 64 GB, 30 min |
| `training/robustez_5cv/run_agregar_5cv_robustez.slurm` | agregación | 2 CPU, 4 GB |
| `training/robustez_5cv/run_export_final_5cv.slurm` | empaquetado de resultados en `tar` | sin GPU |

```bash
sbatch run_preparar_5cv_robustez.slurm
sbatch run_5cv_robustez_array.slurm           # espera a que termine
sbatch run_agregar_5cv_robustez.slurm
```

---

## Buenas prácticas que conviene copiar en tu propio proyecto

1. **Divide por paciente** (`StratifiedGroupKFold(groups=patient_id)`), nunca por imagen.
2. **Congela el split** en un fichero con su hash y haz que el código se detenga si cambia.
3. **Escribe el protocolo de evaluación antes de abrir el test**, guárdalo con su hash y abre el test una vez.
4. **Entrena un clasificador de control sin red neuronal.** Si acierta mucho, sospecha de un atajo (escáner, hospital, formato).
5. **Evalúa por paciente**, que es la unidad que importa, y usa balanced accuracy si las clases están desequilibradas.
6. **Reporta la variabilidad** (validación cruzada) además de la cifra de un único test.
