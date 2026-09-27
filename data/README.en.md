**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# data/

There are **no images** here. Only the files that fix which patient goes into each subset, so that anyone can rebuild exactly the same splits. The images are downloaded from their official source with `training/herramientas/construir_manifest_cheng.py` (see [`docs/en/02_data.md`](../docs/en/02_data.md)).

| File | Content |
|---|---|
| `splits/split_multiclase_cheng.csv` / `.json` | Frozen patient-level split: 233 patients → 163 train / 35 val / 35 test. Columns `patient_id, subset, tumor_type, n_slices`. SHA‑256 hash: `fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092` |
| `splits/robustez_5cv/fold_{1..5}.json` | Training and evaluation patients of each robustness-validation fold (StratifiedGroupKFold, `random_state=20260904`) |
| `splits/robustez_5cv/folds_master.json` / `.sha256` | Index of the 5 folds with the hash of each one |
| `splits/robustez_5cv/distribucion_folds.csv` | Number of patients per class in each fold |
| `manifest_template.csv` | Column schema of the manifest (`manifest_final.csv`) |

The identifiers (`CHENG-100360`…) are the `PID`s published by the Cheng dataset itself (data anonymised by its authors). Subset values: `train`, `val`, `test`; classes: `glioma`, `meningioma`, `pituitario` (Spanish for pituitary).
