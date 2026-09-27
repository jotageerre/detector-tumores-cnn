**Language:** [Español](../07_limitaciones_y_etica.md) · **English** · [Français](../fr/07_limites_et_ethique.md)

# 7. Limitations and ethical considerations

Source: chapters 11 and 13 of the thesis (Spanish).

## What this work does NOT show

- **There is no external validation.** Everything was evaluated within the Cheng cohort (233 patients, two Chinese hospitals, contrast-enhanced T1). How the model behaves with other scanners, protocols, hospitals or populations is unknown.
- **There is no clinical validation.** It is neither a medical device nor a diagnostic tool.
- **The test set is small.** 35 patients: a single misclassified patient moves the balanced accuracy by several points. Hence the very wide confidence interval [84.26 %, 100 %] and the importance of the 5-fold validation.
- **The 5CV is not nested.** The architecture was chosen earlier with a different split; the 5CV measures the variability of the chosen model, not of the whole selection process.
- **Grad-CAM is qualitative.** It was not compared with the tumour masks, so it does not prove that the model looks at the tumour.
- **Possible shortcuts within Cheng.** A low-level classifier reaches 69.22 % on the multiclass task (chance would be 33 %). It has not been shown that the model relies on this, but it has not been ruled out either.
- **No strict GPU determinism.** Re-runs will not give exactly the same figures.
- **Part of the architecture-selection code has not been preserved** (the exact notebook version with the fine-tuning, the ensemble and the final evaluation; Appendix A.3).

## Data

| Aspect | Cheng | IXI |
|---|---|---|
| Anonymisation | Explicitly stated by the authors | Not verified in the same detail |
| Ethical approval | Stated (committees of Nanfang Hospital and General Hospital of Tianjin Medical University) | Not verified in the same detail |
| Informed consent | No explicit mention found | Not verified |
| Licence | CC BY 4.0 (verified with reservations) | CC BY-SA 3.0, attribution required |

This repository **does not redistribute images**: only patient identifiers from the public dataset itself (e.g. `CHENG-100360`), which are needed to reproduce the split.

## The desktop app

- Local user passwords are stored **in plain text** in `credentials.json` (a real limitation of the demo, not a security feature).
- The results (Grad-CAM, 3D viewer) are published in publicly readable buckets.
- There is no persistent log of which image, model and result each inference produced.

For all these reasons, **do not use the app with real patient data**.

## Proposed future work

1. External validation on an independent cohort (another centre, another scanner).
2. Nested cross-validation (model selection inside the estimate).
3. Quantitative evaluation of Grad-CAM against Cheng's tumour masks.
4. Look for residual centre/scanner signals within the cohort and try other patient-level aggregation rules.
5. GPU determinism, a reproducible container environment and complete packaging.
6. In the app: the same preprocessing as in training, traceability of every inference, and private storage.
