**Language:** [Español](../05_guia_reproduccion.md) · **English** · [Français](../fr/05_guide_reproduction.md)

# 5. Step-by-step reproduction guide

This guide explains how to recreate the thesis experiments, from the fastest route to the one most faithful to the original. It summarises and extends Appendix G (reproducibility instructions).

> **What "reproduce" means here.** What is reproduced is the **procedure**: same data, same split, same configuration, same protocol. Identical figures down to the last decimal are not guaranteed, because strict GPU determinism was not enabled (Appendix G.4). Expect results of the same order, not identical ones.

## 0. Requirements

| | Minimum | Used in the thesis |
|---|---|---|
| Python | 3.10 | 3.10.8 |
| TensorFlow | 2.15.1 | 2.15.1 |
| GPU | any NVIDIA with ≥ 8 GB (it works on CPU, but takes hours) | NVIDIA A30 24 GB, CUDA 12.2 |
| Disk | ~3 GB (ZIPs + PNGs) | |
| OS | Linux or **WSL2** on Windows (TensorFlow no longer uses the GPU on native Windows) | Linux (HPC cluster) |

```bash
git clone https://github.com/<your-user>/detector-tumores-cnn.git
cd detector-tumores-cnn/training
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # on Linux with GPU: pip install "tensorflow[and-cuda]==2.15.1"
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

> **scikit-learn < 1.8.** Version 1.8 changed how `StratifiedGroupKFold` shuffles and produces different folds. Versions 1.5.2, 1.6.1 and 1.7.2 reproduce the archived folds byte for byte (checked).

---

## Route A — Fast: prepare the data and train the final model (≈ 30 min with a GPU)

### A.1 Download Cheng and build the manifest

```bash
python herramientas/construir_manifest_cheng.py \
    --root ./tfg_run \
    --split ../data/splits/split_multiclase_cheng.csv
```

What it does: downloads the 4 ZIP files from figshare (~880 MB) checking their MD5, converts the 3,064 `.mat` files to PNG with the original preprocessing function, assigns each patient its subset from the frozen split and **checks that the split hash is `fda7e2da…`**. If anything does not match, it stops.

Expected output at the end:

```
  huella del split : fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092
OK: .../tfg_run/results/artifacts/manifest_final.csv  (3064 cortes, 233 pacientes, split verificado)
tumor_type  glioma  meningioma  pituitario
subset
test            12          10          13
train           63          62          38
val             14          10          11
```

If figshare blocks the automatic download (some networks return 403), download the 4 ZIP files manually from the [dataset page](https://doi.org/10.6084/m9.figshare.1512427), extract the `.mat` files into `tfg_run/data/raw/cheng_mat/` and run the script with `--sin-descarga` ("no download").

### A.2 Train the final model and evaluate it once on the test set

```bash
python entrenar_modelo_final.py --root ./tfg_run
```

- Trains ResNet50‑FT on the 198 train+val patients (6 head epochs + 6 fine-tuning epochs, seed 42).
- Saves `tfg_run/results/tumor_type/final/resnet50_ft_cheng_final.keras` and **`.h5`** (the latter is the one used by the Cloud Run backend).
- Evaluates the 35 test patients **only once** and writes `resultado_evaluacion_final.json`, the per-slice and per-patient predictions, and the `.lock` files. If you run it again in the same folder, it refuses to open the test set a second time.

Thesis reference: balanced accuracy 93.89 %, accuracy 94.29 %, macro-F1 93.98 %.

> This script is a **re-implementation** of protocol C.1: the exact notebook of job 70930 has not been preserved. It reuses, line by line, the recipe in `robustez_5cv/ejecutar_fold_5cv_robustez.py`, which is original code.

---

## Route B — 5-fold robustness validation with the original scripts

The scripts in `training/robustez_5cv/` are **exactly** the ones run on the cluster (jobs 71101, 71102, 71107). They have the cluster path hard-coded; replace it with yours (the script's hash will change, which is expected):

```bash
cd training/robustez_5cv
RUTA=$(realpath ../tfg_run)          # the --root folder from Route A
sed -i "s#/shared/home/FYK3492/TFG_tumores/tfg_run#$RUTA#" preparar_5cv_robustez.py ejecutar_fold_5cv_robustez.py agregar_5cv_robustez.py
# folders that already existed on the cluster and that the scripts take for granted
mkdir -p "$RUTA/frozen/robustez_5cv" "$RUTA/results/diagnostics/robustez_5cv"
# the protocol written before running (the scripts store its SHA-256 in every result)
cp ../../docs/protocolos/C2_validacion_robustez_5cv.md "$RUTA/frozen/robustez_5cv/protocolo_5cv_robustez.md"

python preparar_5cv_robustez.py                    # 1) creates the 5 folds (StratifiedGroupKFold, seed 20260904)
for i in 0 1 2 3 4; do                             # 2) one independent model per fold (seeds 1001..1005)
    python ejecutar_fold_5cv_robustez.py --fold-index $i
done
python agregar_5cv_robustez.py                     # 3) checks coverage of all 233 patients and aggregates
```

Check that your folds are the thesis ones:

```bash
diff <(python -m json.tool $RUTA/results/tumor_type/robustez_5cv/splits/fold_1.json) \
     <(python -m json.tool ../../data/splits/robustez_5cv/fold_1.json) && echo "fold 1 identical"
```

Compare your `resultado_5cv_validacion_robustez.json` with [`results/robustez_5cv/`](../../results/robustez_5cv): reference mean 89.44 % ± 4.64 pp.

---

## Route C — Full original pipeline (notebook)

`training/notebooks/02_pipeline_patient_level_multiclase.ipynb` is the complete pipeline as used for the multiclass benchmark: it downloads Cheng **and IXI**, builds the 5,872-image manifest, detects duplicates, checks the frozen split, reproduces the historical baseline and runs the architecture benchmark. Its comments are in Spanish.

1. Set the working folder: `export TFG_PROJECT_ROOT=$PWD/tfg_run_nb` (on Colab it defaults to `/content/tfg_tumores`).
2. Copy the frozen split: `mkdir -p $TFG_PROJECT_ROOT/frozen && cp ../data/splits/split_multiclase_cheng.* $TFG_PROJECT_ROOT/frozen/`
3. Choose what to run in **Section 1.5** of the notebook:
   - `RUN_MODE = "baseline_only"` → data + split + custom-CNN baseline (regression check).
   - `RUN_MODE = "multiclass_benchmark"` → also the 5 architectures with 3-fold GroupCV (table 4.2).
   - `RUN_MODE = "full"` and `RUN_BINARY_SECONDARY = True` → also the historical binary task with IXI.
4. Run everything: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/02_pipeline_patient_level_multiclase.ipynb`

Warnings:
- The notebook **stops** if the recomputed split does not match the frozen one: this is intentional.
- The IXI download may fail (the Imperial server blocks some IPs); the notebook retries and uses Zenodo as a fallback.
- The baseline **regression check** may stop the run (it happened to job 70886; see [04_results.md](04_results.md#41-historical-baseline-custom-cnn-trained-from-scratch)).
- The fine-tuning and ensemble phases (protocols C.3 and C.4) **are not in this version of the notebook** (Appendix A.3). Their protocols are in [`protocolos/`](../protocolos) for anyone who wants to re-implement them.

`01_reconstruccion_dataset_y_control_3clases.ipynb` is the earlier version of the rebuild stage (job 70436), with outputs; it is kept as a historical reference. See [`training/notebooks/baseline_multiclase_cheng.md`](../../training/notebooks/baseline_multiclase_cheng.md) for how the versions relate.

---

## Route D — On a SLURM cluster

The `.slurm` files are the originals. Adapt them to your cluster (partition, account, modules and paths):

| Script | What it launches | Original resources |
|---|---|---|
| `training/slurm/run_TFG_tumores_cerebrales_patient_level_multiclase.slurm` | notebook 02 via `nbconvert` | template with `<PARTITION>`, `<ACCOUNT>`… |
| `training/robustez_5cv/run_preparar_5cv_robustez.slurm` | fold creation | 2 CPUs, 4 GB, no GPU |
| `training/robustez_5cv/run_5cv_robustez_array.slurm` | the 5 folds as a *job array* (`--array=0-4%1`) | 1× A30, 8 CPUs, 64 GB, 30 min |
| `training/robustez_5cv/run_agregar_5cv_robustez.slurm` | aggregation | 2 CPUs, 4 GB |
| `training/robustez_5cv/run_export_final_5cv.slurm` | packaging results into a `tar` | no GPU |

```bash
sbatch run_preparar_5cv_robustez.slurm
sbatch run_5cv_robustez_array.slurm           # wait until it finishes
sbatch run_agregar_5cv_robustez.slurm
```

---

## Good practices worth copying into your own project

1. **Split by patient** (`StratifiedGroupKFold(groups=patient_id)`), never by image.
2. **Freeze the split** in a file with its hash and make the code stop if it changes.
3. **Write the evaluation protocol before opening the test set**, store it with its hash and open the test set once.
4. **Train a control classifier without a neural network.** If it scores high, suspect a shortcut (scanner, hospital, format).
5. **Evaluate per patient**, the unit that matters, and use balanced accuracy if the classes are imbalanced.
6. **Report the variability** (cross-validation) as well as the figure from a single test set.
