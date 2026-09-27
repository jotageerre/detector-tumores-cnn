**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# results/

Original results of the **5-fold robustness validation** (jobs 71102 and 71107), copied unchanged (field names and texts are in Spanish). The models (`.keras`, ~214 MB each) are not included.

| File | Content |
|---|---|
| `robustez_5cv/protocolo_5cv_robustez.md` | Protocol written before running |
| `robustez_5cv/resultado_5cv_robustez.md` | Summary written after aggregation |
| `robustez_5cv/resultado_5cv_validacion_robustez.json` | Per-fold metrics, mean ± SD and overall metrics over the 233 predictions |
| `robustez_5cv/resultado_5cv_fold_{1..5}.json` | Full result of each fold (training history, unfrozen layers, hashes) |
| `robustez_5cv/resumen_5cv_folds.csv` | One row per fold |
| `robustez_5cv/confusion_matrix_5cv_agregada.csv` | Confusion matrix of the 233 patients |
| `robustez_5cv/predicciones_paciente_5cv_oof_233.csv` | Probabilities and prediction for each patient, evaluated by a model that never saw them |

The results of the architecture selection and of the final test evaluation are in Appendix D ([`docs/en/04_results.md`](../docs/en/04_results.md)); their original files belong to a 16 GB package that is not uploaded to the repository.
