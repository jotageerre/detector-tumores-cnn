**Langue :** [Español](../03_metodologia.md) · [English](../en/03_methodology.md) · **Français**

# 3. Méthodologie d'apprentissage profond

Source : chapitres 7, 8 et 9 du mémoire ; annexes C et H (en espagnol).

<p align="center"><img src="../img/pipeline.jpg" width="60%"></p>

## 3.1 Entrée du modèle

| Étape | Détail |
|---|---|
| Image | PNG 8 bits 256×256 (voir [02_donnees.md](02_donnees.md#23-prétraitement-de-chaque-coupe-identique-pour-tout-le-projet)) |
| Redimensionnement | 224×224, bilinéaire |
| Canaux | Le canal gris est **répliqué** 3 fois (les réseaux ImageNet attendent du RGB ; aucune information n'est ajoutée) |
| Normalisation | `tf.keras.applications.resnet50.preprocess_input` |
| Augmentation de données (entraînement uniquement) | `RandomFlip("horizontal")`, `RandomRotation(0.03)`, `RandomZoom(0.10)`, `RandomTranslation(0.05, 0.05)`, `RandomContrast(0.10)` (code réel de `ejecutar_fold_5cv_robustez.py`) |

Aucune correction de champ de biais, recalage, segmentation préalable, normalisation z-score ni CLAHE n'a été appliquée.

## 3.2 Apprentissage par transfert en deux phases

<p align="center"><img src="../img/transfer_learning.jpg" width="65%"></p>

**Tête commune** à toutes les architectures :

```
backbone (ImageNet, sans top)
 → GlobalAveragePooling2D
 → Dropout(0.4)
 → Dense(128, ReLU)
 → Dropout(0.3)
 → Dense(3, softmax)          # gliome, méningiome, hypophysaire
perte : sparse_categorical_crossentropy · batch 32
```

| | Phase 1 : tête | Phase 2 : ajustement fin |
|---|---|---|
| Backbone | gelé | **20 dernières couches hors BatchNormalization** dégelées |
| BatchNormalization | gelée | **gelée** (vérifié dans les artefacts) |
| Optimiseur | Adam 1e‑4 | Adam 1e‑5 |
| Époques (modèle final et 5CV) | 6 | 6 |
| Paramètres entraînables (ResNet50) | 262 659 | 14 691 331 (sur 23 850 371) |

Le nombre d'époques du modèle final (6 + 6) a été fixé comme la **médiane** des meilleures époques observées dans les plis de sélection ([5, 9, 6] et [7, 3, 6]), avant d'ouvrir le test.

## 3.3 Sélection de l'architecture (sans toucher au test)

- Candidates : **VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3** (familles différentes : classique, résiduelle, efficiente, mise à l'échelle composée, multi-échelle).
- Protocole : validation croisée **groupée par patient** (StratifiedGroupKFold, 3 plis) sur les **198 patients** d'entraînement+validation, avec des prédictions hors pli (OOF). Le test de 35 patients reste verrouillé.
- Phase *gelée* pour les 5 → ajustement fin des 2 meilleures ([protocole C.3](../protocolos/C3_fine_tuning_candidatos.md)) → ensemble 50/50 ([protocole C.4](../protocolos/C4_combinacion_modelos.md)) → gagne celle qui a la plus haute balanced accuracy OOF.

<p align="center"><img src="../img/protocolo.jpg" width="55%"></p>

## 3.4 Évaluation au niveau du patient

Le modèle prédit par coupe, mais il est évalué par **patient** :

```python
# Annexe H.1 — seule règle d'agrégation implémentée
agr = coupes.groupby("patient_id").agg({"p_glioma":"mean", "p_meningioma":"mean", "p_pituitario":"mean"})
pred_patient = agr.values.argmax(1)
balanced_accuracy_score(vrai_patient, pred_patient)   # moyenne du rappel par classe
```

| Métrique | Rôle |
|---|---|
| **Balanced accuracy** (moyenne non pondérée du rappel des 3 classes) | principale |
| Accuracy | secondaire |
| Macro-F1 | secondaire |
| Rappel/précision par classe, matrice de confusion, IC 95 % bootstrap (2 000 rééchantillonnages) | complémentaires |

> Attention aux comparaisons : dans la tâche binaire historique, la « balanced accuracy » valait (sensibilité + spécificité)/2 ; en multiclasse, c'est la moyenne des rappels. Même nom, formule différente.

## 3.5 Explicabilité (Grad-CAM)

Grad-CAM sur la dernière couche convolutive, localisée dynamiquement (`conv5_block3_out` dans ResNet50). Il est utilisé **uniquement comme analyse qualitative a posteriori** : il n'a pas été comparé aux masques tumoraux du jeu de données, il ne prouve donc pas que le modèle « localise » la tumeur. Le backend Cloud Run génère le même type de carte pour chaque prédiction.

## 3.6 Traçabilité et contrôles automatiques

| Mécanisme | Ce qu'il protège |
|---|---|
| `manifest_hash`, `split_hash`, `config_hash` | De quelles données, partition et configuration provient chaque résultat |
| SHA‑256 des checkpoints, protocoles et scripts | Qu'un fichier n'a pas changé |
| `FINAL_TEST_STARTED.lock` / `FINAL_TEST_COMPLETED.lock` | Que le test n'est ouvert qu'une fois, avec le même modèle/protocole/partition |
| `assert` de disjonction des patients et de couverture des 233 | Pas de fuite ni de patient manquant |
| `assert` du nombre de paramètres | Que l'architecture chargée est celle attendue |
| Contrôle de régression (±0,03) par rapport à la référence historique (79,66 %) | Que le framework reconstruit reproduit les résultats antérieurs ; lève `RuntimeError` sinon |

## 3.7 Exécution sur HPC

Chaque expérience est un job SLURM non interactif qui exécute le notebook avec `jupyter nbconvert --execute` (ou un script Python) et conserve ses journaux `.out/.err`. 16 jobs sont documentés (annexe B). Environnement : Python 3.10.8, TensorFlow 2.15.1, CUDA 12.2, NVIDIA A30 24 Go.

```bash
#SBATCH --partition=main
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
```

**Déterminisme :** les graines ont été fixées, mais pas le déterminisme strict du GPU. Reproduire la procédure ne garantit pas des chiffres identiques à la dernière décimale.
