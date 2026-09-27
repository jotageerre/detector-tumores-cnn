**Langue :** [Español](../02_datos.md) · [English](../en/02_data.md) · **Français**

# 2. Données

Ce dépôt **ne contient aucune image médicale**. Il contient ce qu'il faut pour les télécharger depuis leurs sources officielles et reconstruire exactement la même partition par patient.

Source : chapitre 6 du mémoire et annexes A et H des [annexes techniques](../TFG_Joaquin_Gonzalez_Rodriguez_anexos.pdf) (en espagnol).

## 2.1 Jeu de données du résultat final : Cheng et al. (figshare)

| | |
|---|---|
| DOI | [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) |
| Licence | CC BY 4.0 (vérifiée avec réserve, voir [07_limites_et_ethique.md](07_limites_et_ethique.md)) |
| Contenu | 3 064 coupes d'IRM **T1 avec contraste** de **233 patients**, acquises au Nanfang Hospital et au General Hospital de la Tianjin Medical University |
| Format | 4 ZIP contenant des fichiers `.mat` (MATLAB v7.3 / HDF5). Chacun contient `cjdata.label`, `cjdata.PID`, `cjdata.image`, `cjdata.tumorMask`, `cjdata.tumorBorder` |
| Étiquettes | `1` = méningiome, `2` = gliome, `3` = tumeur hypophysaire |

Fichiers et MD5 publiés (le script les vérifie automatiquement) :

| ZIP | file_id | MD5 |
|---|---|---|
| brainTumorDataPublic_1-766.zip | 3381290 | `74b949ad33f042e6e103523091cd1428` |
| brainTumorDataPublic_767-1532.zip | 3381296 | `7e8a875500d2c8a346f270538e29890e` |
| brainTumorDataPublic_1533-2298.zip | 3381293 | `8227bf6080cb71f15a88be8d25c79ae7` |
| brainTumorDataPublic_2299-3064.zip | 3381302 | `b378a80d6174e5317d59eb28430c6652` |

<p align="center"><img src="../img/ejemplos_cheng.jpg" width="75%"></p>

### Répartition

| Classe | Patients | Coupes | Entraînement | Validation | Test |
|---|---|---|---|---|---|
| Gliome | 89 | 1 426 | 63 | 14 | 12 |
| Méningiome | 82 | 708 | 62 | 10 | 10 |
| Hypophysaire | 62 | 930 | 38 | 11 | 13 |
| **Total** | **233** | **3 064** | **163** (2 091 coupes) | **35** (499) | **35** (474) |

Les classes sont déséquilibrées et le nombre de coupes par patient varie beaucoup (de 1 à 38, médiane 13). C'est pourquoi la métrique principale est la **balanced accuracy au niveau du patient**.

## 2.2 Jeu de données IXI (tâche binaire historique uniquement)

Il n'a servi que de classe « sans tumeur » dans la tâche binaire finalement abandonnée à cause du biais de provenance. Licence **CC BY-SA 3.0**. Le notebook `02` essaie de le télécharger depuis le serveur de l'Imperial College et, s'il renvoie 403, se rabat sur [l'enregistrement Zenodo 7047668](https://zenodo.org/records/7047668), qui redistribue des volumes T1 d'IXI avec un MD5 par fichier. **Il n'est pas nécessaire pour reproduire le résultat multiclasse.**

## 2.3 Prétraitement de chaque coupe (identique pour tout le projet)

Implémenté dans la fonction `_a_png` du notebook et copié sans modification dans `training/herramientas/construir_manifest_cheng.py` :

1. Normalisation par les percentiles 1–99 vers l'intervalle [0, 1].
2. **Recadrage sur la boîte englobante du cerveau** : pixels au-dessus de 10 % → boîte carrée centrée avec une marge de 4 %.
3. Redimensionnement à 256×256 (LANCZOS) et enregistrement en PNG 8 bits en niveaux de gris.
4. À l'entraînement : redimensionnement à 224×224, réplication du canal gris sur 3 canaux et `preprocess_input` de ResNet50.

## 2.4 Le manifeste

`manifest_final.csv` est le contrat d'entrée du pipeline : une ligne par image. Schéma des colonnes dans [`data/manifest_template.csv`](../../data/manifest_template.csv) :

`local_file, label, patient_id, patient_id_source, volume_id, slice_id, view, source_dataset, source_url, original_file, site, tumor_type, modality, sha256, phash, width, height, duplicate_cluster, subset`

Le manifeste original n'est pas publié (il contient des chemins du cluster et les lignes IXI). Le script `construir_manifest_cheng.py` en génère un équivalent pour Cheng.

## 2.5 Contrôles de fuite d'information

- **`patient_id` disjoints** entre entraînement, validation et test, et entre les plis de chaque validation croisée.
- **SHA‑256 unique** par image (0 doublon exact parmi les 5 872 images).
- **Quasi-doublons** : pHash de 256 bits (distance de Hamming ≤ 12) + confirmation SSIM ≥ 0,92. Résultat : **0 quasi-doublon confirmé** (ni entre sources, ni entre classes, ni entre patients). Rien n'a jamais été supprimé de façon destructive.

## 2.6 Partition figée et empreintes d'intégrité

La partition par patient se trouve dans [`data/splits/split_multiclase_cheng.csv`](../../data/splits/split_multiclase_cheng.csv) (et `.json`). Son empreinte se calcule ainsi :

```python
pac = cheng.groupby("patient_id").agg(subset=("subset","first"), tumor_type=("tumor_type","first"))
pac = pac.reset_index().sort_values("patient_id")
sha256(pac[["patient_id","subset","tumor_type"]].to_csv(index=False).encode()).hexdigest()
# -> fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092
```

Chaque exécution charge la partition et **s'arrête si l'empreinte ne correspond pas**, au lieu de générer une nouvelle partition. Empreintes documentées (annexe H.3) :

| Artefact | Empreinte |
|---|---|
| Partition multiclasse 163/35/35 | `fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092` |
| Plis 5CV (`folds_master.json`) | `cf13cf8e661f3f010bbe3761ebf3ec09a8c4f9b879aceb55b2f8a169e5945674` |
| Plis 5CV (hash sémantique) | `985d9fc941ac672c9226a434ee31bbd827c2dc063e12e28b0235a5bd2550ce72` |
| Manifeste original complet (5 872 lignes) | `c468eb4e057dfab9b883b73c59e959f6194b0c46037df6b7cef22c9b2e5e6487` |

Les 5 plis de la validation de robustesse se trouvent dans [`data/splits/robustez_5cv/`](../../data/splits/robustez_5cv) et sont régénérés **octet pour octet** par `preparar_5cv_robustez.py` avec scikit-learn 1.5–1.7 (la version 1.8 produit des plis différents ; vérifié lors de la préparation de ce dépôt).
