**Language:** [Español](../02_datos.md) · **English** · [Français](../fr/02_donnees.md)

# 2. Data

This repository **contains no medical images**. It contains what you need to download them from their official sources and rebuild exactly the same patient-level split.

Source: chapter 6 of the thesis and Appendices A and H of the [technical appendices](../TFG_Joaquin_Gonzalez_Rodriguez_anexos.pdf) (Spanish).

## 2.1 Dataset used for the final result: Cheng et al. (figshare)

| | |
|---|---|
| DOI | [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) |
| Licence | CC BY 4.0 (verified with reservations, see [07_limitations_and_ethics.md](07_limitations_and_ethics.md)) |
| Content | 3,064 **contrast-enhanced T1** MRI slices from **233 patients**, acquired at Nanfang Hospital and General Hospital of Tianjin Medical University |
| Format | 4 ZIP files with `.mat` files (MATLAB v7.3 / HDF5). Each contains `cjdata.label`, `cjdata.PID`, `cjdata.image`, `cjdata.tumorMask`, `cjdata.tumorBorder` |
| Labels | `1` = meningioma, `2` = glioma, `3` = pituitary tumour |

Files and published MD5 checksums (the script checks them automatically):

| ZIP | file_id | MD5 |
|---|---|---|
| brainTumorDataPublic_1-766.zip | 3381290 | `74b949ad33f042e6e103523091cd1428` |
| brainTumorDataPublic_767-1532.zip | 3381296 | `7e8a875500d2c8a346f270538e29890e` |
| brainTumorDataPublic_1533-2298.zip | 3381293 | `8227bf6080cb71f15a88be8d25c79ae7` |
| brainTumorDataPublic_2299-3064.zip | 3381302 | `b378a80d6174e5317d59eb28430c6652` |

<p align="center"><img src="../img/ejemplos_cheng.jpg" width="75%"></p>

### Distribution

| Class | Patients | Slices | Train | Val | Test |
|---|---|---|---|---|---|
| Glioma | 89 | 1,426 | 63 | 14 | 12 |
| Meningioma | 82 | 708 | 62 | 10 | 10 |
| Pituitary | 62 | 930 | 38 | 11 | 13 |
| **Total** | **233** | **3,064** | **163** (2,091 slices) | **35** (499) | **35** (474) |

The classes are imbalanced and the number of slices per patient varies a lot (from 1 to 38, median 13). That is why the main metric is **patient-level balanced accuracy**.

## 2.2 IXI dataset (historical binary task only)

It was used only as the "no tumour" class in the binary task that was eventually dropped because of source bias. Licence **CC BY-SA 3.0**. Notebook `02` tries to download it from the Imperial College server and, if that returns 403, falls back to [Zenodo record 7047668](https://zenodo.org/records/7047668), which redistributes IXI T1 volumes with a per-file MD5. **It is not needed to reproduce the multiclass result.**

## 2.3 Preprocessing of each slice (same for the whole project)

Implemented in the `_a_png` function of the notebook and copied unchanged into `training/herramientas/construir_manifest_cheng.py`:

1. Normalisation by the 1st–99th percentiles to the [0, 1] range.
2. **Crop to the brain bounding box**: pixels above 10 % → square box centred with a 4 % margin.
3. Resize to 256×256 (LANCZOS) and save as an 8-bit greyscale PNG.
4. During training: resize to 224×224, replicate the grey channel into 3 channels and apply ResNet50's `preprocess_input`.

## 2.4 The manifest

`manifest_final.csv` is the pipeline's input contract: one row per image. Column schema in [`data/manifest_template.csv`](../../data/manifest_template.csv):

`local_file, label, patient_id, patient_id_source, volume_id, slice_id, view, source_dataset, source_url, original_file, site, tumor_type, modality, sha256, phash, width, height, duplicate_cluster, subset`

The original manifest is not published (it contains cluster paths and the IXI rows). The `construir_manifest_cheng.py` script generates an equivalent one for Cheng.

## 2.5 Leakage controls

- **Disjoint `patient_id`** across train, val and test, and across folds of every cross-validation.
- **Unique SHA‑256** per image (0 exact duplicates among the 5,872 images).
- **Near-duplicates**: 256-bit pHash (Hamming distance ≤ 12) + SSIM confirmation ≥ 0.92. Result: **0 confirmed near-duplicates** (neither across sources, nor across classes, nor across patients). Nothing was ever deleted destructively.

## 2.6 Frozen split and integrity hashes

The patient-level split is in [`data/splits/split_multiclase_cheng.csv`](../../data/splits/split_multiclase_cheng.csv) (and `.json`). Its hash is computed like this:

```python
pac = cheng.groupby("patient_id").agg(subset=("subset","first"), tumor_type=("tumor_type","first"))
pac = pac.reset_index().sort_values("patient_id")
sha256(pac[["patient_id","subset","tumor_type"]].to_csv(index=False).encode()).hexdigest()
# -> fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092
```

Every run loads the split and **stops if the hash does not match**, instead of generating a new split. Documented hashes (Appendix H.3):

| Artefact | Hash |
|---|---|
| Multiclass split 163/35/35 | `fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092` |
| 5CV folds (`folds_master.json`) | `cf13cf8e661f3f010bbe3761ebf3ec09a8c4f9b879aceb55b2f8a169e5945674` |
| 5CV folds (semantic hash) | `985d9fc941ac672c9226a434ee31bbd827c2dc063e12e28b0235a5bd2550ce72` |
| Full original manifest (5,872 rows) | `c468eb4e057dfab9b883b73c59e959f6194b0c46037df6b7cef22c9b2e5e6487` |

The 5 folds of the robustness validation are in [`data/splits/robustez_5cv/`](../../data/splits/robustez_5cv) and are regenerated **byte for byte** by `preparar_5cv_robustez.py` with scikit-learn 1.5–1.7 (version 1.8 produces different folds; checked while preparing this repository).
