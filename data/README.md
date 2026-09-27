# data/

Aquí **no hay imágenes**. Solo los ficheros que fijan qué paciente va a cada subconjunto, para que cualquiera pueda reconstruir exactamente los mismos repartos. Las imágenes se descargan de su fuente oficial con `training/herramientas/construir_manifest_cheng.py` (ver [`docs/02_datos.md`](../docs/02_datos.md)).

| Fichero | Contenido |
|---|---|
| `splits/split_multiclase_cheng.csv` / `.json` | Split congelado por paciente: 233 pacientes → 163 train / 35 val / 35 test. Columnas `patient_id, subset, tumor_type, n_slices`. Huella SHA‑256: `fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092` |
| `splits/robustez_5cv/fold_{1..5}.json` | Pacientes de entrenamiento y evaluación de cada fold de la validación de robustez (StratifiedGroupKFold, `random_state=20260904`) |
| `splits/robustez_5cv/folds_master.json` / `.sha256` | Índice de los 5 folds con la huella de cada uno |
| `splits/robustez_5cv/distribucion_folds.csv` | Nº de pacientes por clase en cada fold |
| `manifest_template.csv` | Esquema de columnas del manifiesto (`manifest_final.csv`) |

Los identificadores (`CHENG-100360`…) son los `PID` que publica el propio dataset de Cheng (datos anonimizados por sus autores).
