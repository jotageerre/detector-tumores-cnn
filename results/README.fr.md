**Langue :** [Español](README.md) · [English](README.en.md) · **Français**

# results/

Résultats originaux de la **validation de robustesse à 5 plis** (jobs 71102 et 71107), copiés sans modification (noms de champs et textes en espagnol). Les modèles (`.keras`, ~214 Mo chacun) ne sont pas inclus.

| Fichier | Contenu |
|---|---|
| `robustez_5cv/protocolo_5cv_robustez.md` | Protocole rédigé avant l'exécution |
| `robustez_5cv/resultado_5cv_robustez.md` | Résumé rédigé après l'agrégation |
| `robustez_5cv/resultado_5cv_validacion_robustez.json` | Métriques par pli, moyenne ± ET et métriques globales sur les 233 prédictions |
| `robustez_5cv/resultado_5cv_fold_{1..5}.json` | Résultat complet de chaque pli (historique d'entraînement, couches dégelées, empreintes) |
| `robustez_5cv/resumen_5cv_folds.csv` | Une ligne par pli |
| `robustez_5cv/confusion_matrix_5cv_agregada.csv` | Matrice de confusion des 233 patients |
| `robustez_5cv/predicciones_paciente_5cv_oof_233.csv` | Probabilités et prédiction de chaque patient, évalué par un modèle qui ne l'a jamais vu |

Les résultats de la sélection d'architecture et de l'évaluation finale sur le test figurent dans l'annexe D ([`docs/fr/04_resultats.md`](../docs/fr/04_resultats.md)) ; leurs fichiers originaux font partie d'une archive de 16 Go qui n'est pas envoyée sur le dépôt.
