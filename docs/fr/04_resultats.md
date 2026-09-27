**Langue :** [Español](../04_resultados.md) · [English](../en/04_results.md) · **Français**

# 4. Résultats

Tous les chiffres proviennent de l'annexe D (fichiers de résultats vérifiés). Tous sont **au niveau du patient**.

## 4.1 Référence historique (CNN maison entraîné de zéro)

| Expérience | Balanced acc. | Ensemble |
|---|---|---|
| CNN maison (historique) | 79,66 % | test historique (35 patients) |
| Reproduction avec le framework reconstruit | 74,44 % | même test (contrôle de régression) |

L'écart (−5,21 pts) a dépassé la tolérance de ±0,03 et **le contrôle de régression a automatiquement arrêté** l'une des exécutions (job 70886). C'est documenté comme une limite.

## 4.2 Comparaison des architectures (backbone gelé)

3 plis groupés par patient sur 198 patients · test verrouillé.

| Modèle | Balanced acc. OOF | Accuracy | Macro-F1 | Rappel gliome | Rappel ménin. | Rappel hypoph. | Paramètres |
|---|---|---|---|---|---|---|---|
| **ResNet50** | **83,13 %** | 81,31 % | 82,06 % | 79,22 % | 72,22 % | 97,96 % | 23,9 M |
| InceptionV3 | 82,06 % | 80,30 % | 80,40 % | 85,71 % | 62,50 % | 97,96 % | 22,1 M |
| MobileNetV2 | 81,53 % | 79,29 % | 79,45 % | 77,92 % | 66,67 % | 100,00 % | 2,4 M |
| VGG16 | 78,94 % | 76,77 % | 75,98 % | 89,61 % | 47,22 % | 100,00 % | 14,8 M |
| EfficientNetB0 | 78,72 % | 76,26 % | 75,90 % | 79,22 % | 56,94 % | 100,00 % | 4,2 M |

## 4.3 Ajustement fin et ensemble

| Modèle | Balanced acc. OOF | Accuracy | Macro-F1 |
|---|---|---|---|
| **ResNet50 avec ajustement fin** | **88,54 %** | 87,37 % | 88,02 % |
| Ensemble 50/50 (ResNet50‑FT + InceptionV3‑FT) | 88,29 % | 87,37 % | 87,92 % |
| InceptionV3 avec ajustement fin | 85,76 % | 84,34 % | 84,85 % |

L'ensemble ne dépasse pas le meilleur modèle individuel → **ResNet50‑FT** est sélectionné.

<p align="center"><img src="../img/seleccion_arquitectura.jpg" width="60%"></p>

## 4.4 Évaluation finale sur le test réservé (une seule fois)

| Métrique | Valeur |
|---|---|
| Patients / coupes | 35 / 474 |
| **Balanced accuracy** | **93,89 %** |
| Accuracy | 94,29 % |
| Macro-F1 | 93,98 % |
| Rappel gliome / méningiome / hypophysaire | 91,67 % / 90,00 % / 100,00 % |
| Précision gliome / méningiome / hypophysaire | 100,00 % / 90,00 % / 92,86 % |
| IC 95 % bootstrap (2 000 rééchantillonnages) de la balanced acc. | [84,26 %, 100,00 %] |

Matrice de confusion (lignes = vrai, colonnes = prédit ; gliome / méningiome / hypophysaire) :

```
[[11, 1, 0],
 [ 0, 9, 1],
 [ 0, 0, 13]]
```

33 patients sur 35 correctement classés. Erreurs : un méningiome prédit comme hypophysaire (confiance 0,82, 8 coupes) et un gliome prédit comme méningiome (confiance 0,46, la plus basse du test, 2 coupes).

## 4.5 Validation croisée de robustesse (5 plis, 233 patients)

Réalisée **après** avoir figé le modèle, sans revenir sur la sélection. Un nouveau modèle par pli, à partir des poids ImageNet. Fichiers dans [`results/robustez_5cv/`](../../results/robustez_5cv).

| Pli | Graine | Patients entr. / éval. | Balanced acc. | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| 1 | 1001 | 186 / 47 | 92,42 % | 89,36 % | 90,11 % |
| 2 | 1002 | 186 / 47 | 82,30 % | 80,85 % | 80,32 % |
| 3 | 1003 | 187 / 46 | 94,34 % | 93,48 % | 94,29 % |
| 4 | 1004 | 186 / 47 | 90,07 % | 89,36 % | 89,43 % |
| 5 | 1005 | 187 / 46 | 88,07 % | 86,96 % | 87,07 % |
| **Moyenne ± ET** | | | **89,44 % ± 4,64 pts** | 88,00 % ± 4,63 pts | 88,24 % ± 5,14 pts |

Sur les **233 prédictions agrégées** (chaque patient évalué une fois par un modèle qui ne l'a jamais vu) : balanced accuracy **88,68 %**, accuracy 87,98 %, macro-F1 88,21 %, 205/233 patients corrects.

```
[[79,  9,  1],
 [ 9, 66,  7],
 [ 1,  1, 60]]
```

<p align="center"><img src="../img/folds_5cv.jpg" width="45%"> <img src="../img/holdout_vs_5cv.jpg" width="45%"></p>

**Comment le lire :** les 93,89 % du test se situent près du haut de la fourchette des plis (82,30 %–94,34 %). L'estimation la plus prudente de la performance au sein de cette cohorte est celle de la 5CV. La classe la plus difficile est le **méningiome** (confondu avec l'hypophysaire, surtout dans le pli 2).

## 4.6 Grad-CAM

<p align="center"><img src="../img/gradcam.jpg" width="55%"></p>

Cinq cas du test : trois réussites (une par classe) et les deux erreurs. Dans les réussites, la confiance est > 0,90 ; dans l'une des erreurs, le modèle se trompe avec une confiance de 0,82, ce qui rappelle qu'**une confiance élevée ne signifie pas une bonne réponse**. Galerie complète dans l'annexe E.
