**Langue :** [Español](README.md) · [English](README.en.md) · **Français**

# data/

Il n'y a **aucune image** ici. Seulement les fichiers qui fixent quel patient va dans chaque sous-ensemble, pour que n'importe qui puisse reconstruire exactement les mêmes partitions. Les images se téléchargent depuis leur source officielle avec `training/herramientas/construir_manifest_cheng.py` (voir [`docs/fr/02_donnees.md`](../docs/fr/02_donnees.md)).

| Fichier | Contenu |
|---|---|
| `splits/split_multiclase_cheng.csv` / `.json` | Partition figée par patient : 233 patients → 163 entraînement / 35 validation / 35 test. Colonnes `patient_id, subset, tumor_type, n_slices`. Empreinte SHA‑256 : `fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092` |
| `splits/robustez_5cv/fold_{1..5}.json` | Patients d'entraînement et d'évaluation de chaque pli de la validation de robustesse (StratifiedGroupKFold, `random_state=20260904`) |
| `splits/robustez_5cv/folds_master.json` / `.sha256` | Index des 5 plis avec l'empreinte de chacun |
| `splits/robustez_5cv/distribucion_folds.csv` | Nombre de patients par classe dans chaque pli |
| `manifest_template.csv` | Schéma des colonnes du manifeste (`manifest_final.csv`) |

Les identifiants (`CHENG-100360`…) sont les `PID` publiés par le jeu de données de Cheng lui-même (données anonymisées par ses auteurs). Valeurs de sous-ensemble : `train`, `val`, `test` ; classes : `glioma`, `meningioma`, `pituitario` (hypophysaire en espagnol).
