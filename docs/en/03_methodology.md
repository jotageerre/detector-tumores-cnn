**Language:** [Español](../03_metodologia.md) · **English** · [Français](../fr/03_methodologie.md)

# 3. Deep Learning methodology

Source: chapters 7, 8 and 9 of the thesis; Appendices C and H (Spanish).

<p align="center"><img src="../img/pipeline.jpg" width="60%"></p>

## 3.1 Model input

| Step | Detail |
|---|---|
| Image | 8-bit PNG 256×256 (see [02_data.md](02_data.md#23-preprocessing-of-each-slice-same-for-the-whole-project)) |
| Resize | 224×224, bilinear |
| Channels | The grey channel is **replicated** 3 times (ImageNet networks expect RGB; no information is added) |
| Normalisation | `tf.keras.applications.resnet50.preprocess_input` |
| Data augmentation (training only) | `RandomFlip("horizontal")`, `RandomRotation(0.03)`, `RandomZoom(0.10)`, `RandomTranslation(0.05, 0.05)`, `RandomContrast(0.10)` (actual code in `ejecutar_fold_5cv_robustez.py`) |

No bias-field correction, registration, prior segmentation, z-score or CLAHE was applied.

## 3.2 Two-phase transfer learning

<p align="center"><img src="../img/transfer_learning.jpg" width="65%"></p>

**Common head** for every architecture:

```
backbone (ImageNet, no top)
 → GlobalAveragePooling2D
 → Dropout(0.4)
 → Dense(128, ReLU)
 → Dropout(0.3)
 → Dense(3, softmax)          # glioma, meningioma, pituitary
loss: sparse_categorical_crossentropy · batch 32
```

| | Phase 1: head | Phase 2: fine-tuning |
|---|---|---|
| Backbone | frozen | **last 20 non-BatchNormalization layers** unfrozen |
| BatchNormalization | frozen | **frozen** (verified in the artefacts) |
| Optimiser | Adam 1e‑4 | Adam 1e‑5 |
| Epochs (final model and 5CV) | 6 | 6 |
| Trainable parameters (ResNet50) | 262,659 | 14,691,331 (of 23,850,371) |

The number of epochs for the final model (6 + 6) was set as the **median** of the best epochs observed in the selection folds ([5, 9, 6] and [7, 3, 6]), before opening the test set.

## 3.3 Architecture selection (without touching the test set)

- Candidates: **VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3** (different families: classic, residual, efficient, compound scaling, multi-scale).
- Protocol: **patient-grouped** cross-validation (StratifiedGroupKFold, 3 folds) on the **198 patients** of train+val, with out-of-fold (OOF) predictions. The 35-patient test set stays locked.
- *Frozen* phase for all 5 → fine-tuning of the best 2 ([protocol C.3](../protocolos/C3_fine_tuning_candidatos.md)) → 50/50 ensemble ([protocol C.4](../protocolos/C4_combinacion_modelos.md)) → the one with the highest OOF balanced accuracy wins.

<p align="center"><img src="../img/protocolo.jpg" width="55%"></p>

## 3.4 Patient-level evaluation

The model predicts per slice, but it is evaluated per **patient**:

```python
# Appendix H.1 — the only aggregation rule implemented
agr = slices.groupby("patient_id").agg({"p_glioma":"mean", "p_meningioma":"mean", "p_pituitario":"mean"})
patient_pred = agr.values.argmax(1)
balanced_accuracy_score(patient_true, patient_pred)   # mean of per-class recall
```

| Metric | Role |
|---|---|
| **Balanced accuracy** (unweighted mean of the recall of the 3 classes) | main |
| Accuracy | secondary |
| Macro-F1 | secondary |
| Per-class recall/precision, confusion matrix, 95 % bootstrap CI (2,000 resamples) | complementary |

> Careful when comparing: in the historical binary task "balanced accuracy" was (sensitivity + specificity)/2; in multiclass it is the mean of the recalls. Same name, different formula.

## 3.5 Explainability (Grad-CAM)

Grad-CAM on the last convolutional layer, located dynamically (`conv5_block3_out` in ResNet50). It is used **only as a qualitative post-hoc analysis**: it was not compared with the tumour masks in the dataset, so it does not prove the model "localises" the tumour. The Cloud Run backend generates the same kind of map for every prediction.

## 3.6 Traceability and automatic checks

| Mechanism | What it protects |
|---|---|
| `manifest_hash`, `split_hash`, `config_hash` | Which data, split and configuration each result comes from |
| SHA‑256 of checkpoints, protocols and scripts | That a file has not changed |
| `FINAL_TEST_STARTED.lock` / `FINAL_TEST_COMPLETED.lock` | That the test set is opened only once, with the same model/protocol/split |
| `assert` on patient disjointness and coverage of all 233 | No leakage and no missing patients |
| `assert` on the number of parameters | That the loaded architecture is the expected one |
| Regression check (±0.03) against the historical baseline (79.66 %) | That the rebuilt framework reproduces earlier results; raises `RuntimeError` otherwise |

## 3.7 Running on HPC

Each experiment is a non-interactive SLURM job that runs the notebook with `jupyter nbconvert --execute` (or a Python script) and keeps its `.out/.err` logs. 16 jobs are documented (Appendix B). Environment: Python 3.10.8, TensorFlow 2.15.1, CUDA 12.2, NVIDIA A30 24 GB.

```bash
#SBATCH --partition=main
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
```

**Determinism:** seeds were fixed, but strict GPU determinism was not enabled. Reproducing the procedure does not guarantee identical figures down to the last decimal.
