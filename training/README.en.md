**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# training/

Code to prepare the data, train and evaluate. The full guide, with expected outputs, is in [`docs/en/05_reproduction_guide.md`](../docs/en/05_reproduction_guide.md). Code comments and console messages are in Spanish.

| Path | What it is | Origin |
|---|---|---|
| `requirements.txt` | Environment (TensorFlow 2.15.1, scikit-learn < 1.8) | added |
| `herramientas/construir_manifest_cheng.py` | Downloads Cheng from figshare, generates the PNGs with the original preprocessing and the `manifest_final.csv` with the frozen split (checks the `fda7e2da…` hash) | added |
| `entrenar_modelo_final.py` | Re-implementation of protocol C.1: trains ResNet50‑FT on 198 patients and evaluates the 35 test patients once; exports `.keras` and `.h5` | added |
| `notebooks/02_pipeline_patient_level_multiclase.ipynb` | Full pipeline: downloads Cheng + IXI, manifest, duplicates, frozen split, baseline, 5-architecture benchmark, historical binary task | **original** |
| `notebooks/01_reconstruccion_dataset_y_control_3clases.ipynb` | Version of the rebuild stage (job 70436), with outputs | **original** |
| `notebooks/baseline_multiclase_cheng.md` | Document that froze the multiclass split and the 79.7 % baseline | **original** |
| `robustez_5cv/*.py`, `robustez_5cv/*.slurm` | 5-fold cross-validation exactly as it was run (jobs 71101, 71102, 71107) | **original** |
| `slurm/run_TFG_…_multiclase.slurm` | SLURM launcher for notebook 02 | **original** |

The original scripts have the cluster path hard-coded (`/shared/home/…/tfg_run`). To use them, replace it with yours (the guide gives the `sed` command). They have not been modified here because every result stores the SHA‑256 of the script that produced it.

Quick start:

```bash
pip install -r requirements.txt
python herramientas/construir_manifest_cheng.py --root ./tfg_run --split ../data/splits/split_multiclase_cheng.csv
python entrenar_modelo_final.py --root ./tfg_run
```

Both added scripts have been tested end to end with synthetic data that mimics Cheng's structure (same list of patients and slices): the manifest reproduces the split hash, and training, `.h5` export and evaluation all work. They could not be run with the real data and weights from the environment where the repository was prepared.
