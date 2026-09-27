**Language:** [Español](../01_historia_del_proyecto.md) · **English** · [Français](../fr/01_historique_du_projet.md)

# 1. Project history: what went wrong and how it was fixed

This thesis did not follow a linear plan. Its most useful result for other people is precisely the path it took: three methodological decisions that changed the project and that anyone working with medical images should know about.

Source: chapters 1, 4 and 6 of the [thesis report](../TFG_Joaquin_Gonzalez_Rodriguez_memoria.pdf) (Spanish).

<p align="center"><img src="../img/evolucion_datos.jpg" width="60%"></p>

## Stage 1 — Prototype with Kaggle data (Sep 2025 – Jul 2026)

The first versions of the classifier were trained on MRI collections aggregated and redistributed on Kaggle, organised in one folder per class. They served to learn the problem, build the training infrastructure and try a first interface (the *ScanIA* prototype, see [`legacy/scania`](../../legacy/scania)).

**The problem:** none of those sources kept the patient identifier. The pipeline used a `group_id` taken from the file name as if it were the patient, but there was no way to check it. With that organisation it was **impossible to guarantee** that slices from the same patient did not appear in training and test at the same time.

> No information leakage was demonstrated; what was detected was a **structural risk** that made it impossible to rule it out. That difference matters, and it is what led to rebuilding the data instead of patching it.

## Stage 2 — Rebuild with real patient identifiers (Aug 2026)

The dataset was rebuilt from the **primary sources**, which do publish who each patient is:

| Class | Source | Patient identifier | Licence |
|---|---|---|---|
| tumour | Cheng et al. — figshare | `cjdata.PID` field inside each `.mat` file | CC BY 4.0 |
| no tumour | IXI — Imperial College London | subject ID in the NIfTI file name (`IXI002-Guys-0828-T1`) | CC BY-SA 3.0 |

Result: **467 real patients** (233 Cheng + 234 IXI) and 5,872 images, with a manifest (`manifest_final.csv`) recording each image's patient, source, SHA‑256 and subset. Automatic checks were added: no patient in two subsets, unique SHA‑256 hashes, and near-duplicate detection (pHash + SSIM). The whole run was automated on the University of Seville HPC cluster (job 70436, 28 min on an NVIDIA A30).

## Stage 3 — Source bias (and why the binary task was dropped)

On the rebuilt dataset, the binary "tumour / no tumour" task reached a balanced accuracy of **98.57 %**. It looked like a success. It was not:

- Every tumour came from Cheng (T1 **with** contrast, two Chinese hospitals).
- Every healthy subject came from IXI (T1 **without** contrast, London hospitals).
- **The label matched the source dataset exactly.**

To check this, a control classifier was trained with **only 10 statistical and texture features, with no neural network at all**: it reached **98.16 %** accuracy and 99.55 % AUC. If something that simple separates the classes, the network does not need to "see" the tumour: recognising the hospital is enough. Another control confirmed it: a classifier was able to identify the **acquisition site within IXI** (all healthy subjects) with 89.32 % accuracy.

> **Lesson:** if your positive and negative classes come from different datasets, your model may be learning the scanner, not the disease. Always build a "dumb" control classifier before celebrating.

## Stage 4 — Refocus on multiclass classification within a single cohort

The core of the thesis became distinguishing **glioma, meningioma and pituitary tumour** within Cheng. Since all three classes come from the same cohort, the deterministic link between class and dataset disappears. Not *every* possible shortcut disappears: the same low-level control reaches 69.22 % on this task (above the 33 % chance level), so there is also low-level signal within Cheng. This is documented as a limitation, not hidden.

A patient-level split **163 / 35 / 35** (train / val / test) was frozen with hash `fda7e2daeb9ec2de…`. The 35 test patients stayed **locked** until the very end.

## Stage 5 — Architecture selection, single test evaluation and 5CV (Sep 2026)

1. **Benchmark** of 5 pre-trained architectures (VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3) with 3-fold patient-grouped cross-validation on the 198 train+val patients.
2. **Fine-tuning** of the best two (ResNet50, InceptionV3) and a 50/50 **ensemble**, each step with its protocol written down *before* running it ([`protocolos/`](../protocolos), Spanish).
3. **Final evaluation protocol** written and hashed → final training → **test evaluated only once**: 93.89 % balanced accuracy.
4. **5-fold cross-validation** on the 233 patients to measure variability: 89.44 % ± 4.64 pp. The test result lies near the top of the range, so the 5CV figure is the more cautious estimate.

## Stage 6 — Cloud service and desktop application (Sep 2026)

The final model is served by a **Cloud Run** service (`run-model`), and a Tkinter desktop application lets you upload an image or a NIfTI volume and see the prediction, the Grad-CAM map and a 3D viewer. A real production failure (503 errors caused by running out of memory while loading several copies of the model) was diagnosed and fixed; it is documented in [06_google_cloud_deployment.md](06_google_cloud_deployment.md#known-issue-503-error).

## Effort

306 h logged over 101 sessions (Clockify). The work package that deviated most from the initial estimate was **data and traceability** (63 h versus 25 h planned, +152 %): rebuilding the dataset was not planned, and it turned out to be the most valuable part of the work.
