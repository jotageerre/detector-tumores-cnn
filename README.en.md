**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# Multiclass brain tumour classification in MRI with Deep Learning

**Bachelor's Thesis (TFG)** · BSc in Computer Engineering – Software Engineering · University of Seville (ETSII), 2026
Author: **Joaquín González Rodríguez** · Supervisor: José Cristóbal Riquelme Santos · Co-supervisor: Manuel Carranza García

This repository gathers **everything needed to understand, reproduce and reuse** the thesis: the training pipeline with patient-level traceability, the frozen splits, the experimental protocols, the results, an inference service on **Google Cloud Run** and a **desktop application** that uses it.

> ⚠️ **For experimental and educational use only.** The model was evaluated within a single public cohort (Cheng et al., 233 patients), **with no external or clinical validation**. It is not a medical device and must not be used to diagnose anyone.

> The thesis report and appendices (PDF), the experimental protocols and the code comments are in **Spanish**. The guides in [`docs/en/`](docs/en) summarise all of them in English.

---

## In 30 seconds

| | |
|---|---|
| **Task** | Classify a contrast-enhanced T1 MRI slice as **glioma**, **meningioma** or **pituitary tumour** |
| **Data** | [Cheng et al. (figshare)](https://doi.org/10.6084/m9.figshare.1512427) · 233 patients · 3,064 slices · CC BY 4.0 |
| **Final model** | **ResNet50** pre-trained on ImageNet + fine-tuning of the last 20 layers (ResNet50‑FT) |
| **Evaluation unit** | The **patient** (mean of the slice probabilities + argmax) |
| **Held-out test (35 patients, evaluated once)** | Balanced accuracy **93.89 %** · Accuracy 94.29 % · Macro‑F1 93.98 % · 95 % bootstrap CI [84.26 %, 100 %] |
| **5-fold cross-validation (233 patients)** | Balanced accuracy **89.44 % ± 4.64 pp** across folds · 88.68 % over the 233 pooled predictions |
| **Demo** | [Video (2 min, ES/EN subtitles)](https://youtu.be/QgTOD3KOTpI) |

<p align="center"><img src="docs/img/app_resultado.jpg" width="48%"> <img src="docs/img/gradcam.jpg" width="40%"></p>

## Why this project may be useful to you

Beyond the performance figure, the value of this thesis lies in **how** it was reached. If you are going to train a CNN on medical images, these are the mistakes the project found (and fixed) along the way:

1. **Splitting by image instead of by patient.** Aggregated Kaggle datasets carry no patient identifier: slices from the same patient can end up in both train and test. Here the dataset was rebuilt from the primary sources using the real `PID` stored in each file.
2. **Source bias.** The binary "tumour / no tumour" task combined Cheng (tumour) and IXI (healthy). A classifier using **only 10 low-level features, with no neural network**, reached **98.16 %** accuracy: the label matched the hospital of origin. That is why the project was refocused on multiclass classification within a single cohort.
3. **Opening the test set more than once.** Here the evaluation protocol was written down and frozen with its hash *before* opening the test set, and the test set was evaluated only once.

All of this is explained step by step in [`docs/en/01_project_history.md`](docs/en/01_project_history.md).

---

## Repository structure

```
detector-tumores-cnn/
├── docs/                      Documentation: full thesis and appendices (PDF, Spanish) + guides
│   ├── *.md                          Guides in Spanish
│   ├── en/                           Guides in English
│   │   ├── 01_project_history.md          Evolution, problems found and decisions
│   │   ├── 02_data.md                     Sources, licences, manifest, duplicates, split
│   │   ├── 03_methodology.md              Preprocessing, architectures, training, metrics, Grad-CAM
│   │   ├── 04_results.md                  All result tables
│   │   ├── 05_reproduction_guide.md       ★ How to recreate the training step by step
│   │   ├── 06_google_cloud_deployment.md  ★ How to set up your own Cloud Run + desktop app
│   │   └── 07_limitations_and_ethics.md
│   ├── fr/                           Guides in French
│   └── protocolos/                   Protocols C.1–C.4 exactly as frozen before each experiment
├── training/                  Training and evaluation code
│   ├── notebooks/             Original pipeline notebooks (Colab / HPC)
│   ├── robustez_5cv/          Original scripts and SLURM files of the 5-fold cross-validation
│   ├── slurm/                 SLURM script for the multiclass benchmark
│   ├── herramientas/          construir_manifest_cheng.py (added for the repo)
│   └── entrenar_modelo_final.py      Re-implementation of protocol C.1 (added for the repo)
├── data/                      Frozen patient-level split + 5CV folds (no images)
├── results/                   5CV validation results (JSON/CSV)
├── cloud-run-backend/         Flask inference API (run-model service)
├── desktop-client/            Tkinter desktop application
└── legacy/scania/             Historical interface prototype (reference only)
```

## Getting started

**I just want to understand the work** → read [`docs/en/01_project_history.md`](docs/en/01_project_history.md) and [`docs/en/04_results.md`](docs/en/04_results.md), or the full thesis in [`docs/TFG_Joaquin_Gonzalez_Rodriguez_memoria.pdf`](docs/TFG_Joaquin_Gonzalez_Rodriguez_memoria.pdf) (Spanish).

**I want to retrain the model** → [`docs/en/05_reproduction_guide.md`](docs/en/05_reproduction_guide.md). Short version:

```bash
cd training
pip install -r requirements.txt            # Python 3.10, TensorFlow 2.15.1
python herramientas/construir_manifest_cheng.py --root ./tfg_run --split ../data/splits/split_multiclase_cheng.csv
python entrenar_modelo_final.py --root ./tfg_run   # a few minutes on an A30-class GPU; hours on CPU
```

**I want to deploy the API and use the app** → [`docs/en/06_google_cloud_deployment.md`](docs/en/06_google_cloud_deployment.md).

## Trained models

The weights are not stored in git (the final model is ~170 MB and GitHub does not accept files over 100 MB). They are published in the repository's **Releases** section:

| File | Model |
|---|---|
| `resnet50_ft_cheng_final.h5` | Multiclass ResNet50‑FT served by the `run-model` service |
| `brain_tumor_cnn.h5` | Historical binary CNN (affected by source bias; demo only) |

## What is original and what was added

So that nobody confuses what produced the thesis results with what was prepared afterwards for publication:

| Type | Files |
|---|---|
| **Original thesis artefacts** (unchanged) | `training/notebooks/*.ipynb`, `training/robustez_5cv/*`, `training/slurm/*`, `data/splits/*`, `results/robustez_5cv/*`, `docs/protocolos/*`, `docs/*.pdf`, `legacy/scania/*`, `cloud-run-backend/main.py` |
| **Added for the repository** | `training/herramientas/construir_manifest_cheng.py`, `training/entrenar_modelo_final.py`, all `README*.md` files and guides in `docs/`, `desktop-client/main.py` (only its configuration changed: relative instead of absolute paths, and URL/buckets configurable through environment variables), `cloud-run-backend/Procfile` |

The exact notebook that ran the fine-tuning and the final evaluation (jobs 70929–70931) **has not been preserved** (Appendix A.3). `entrenar_modelo_final.py` rebuilds that protocol from protocol C.1 and from the real 5CV validation script, which uses exactly the same recipe.

## How to cite

```
González Rodríguez, J. (2026). Clasificación multiclase de tumores cerebrales en imágenes de
resonancia magnética mediante aprendizaje profundo [Bachelor's thesis]. Universidad de Sevilla.
```

If you use the data, also cite the original source: Cheng, J. (2024). *Brain Tumor Dataset.* figshare. https://doi.org/10.6084/m9.figshare.1512427.v8 (CC BY 4.0), together with the article the authors ask to be cited on the dataset page.
