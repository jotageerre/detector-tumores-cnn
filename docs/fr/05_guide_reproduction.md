**Langue :** [Español](../05_guia_reproduccion.md) · [English](../en/05_reproduction_guide.md) · **Français**

# 5. Guide de reproduction pas à pas

Ce guide explique comment recréer les expériences du TFG, de la voie la plus rapide à la plus fidèle à l'original. Il résume et complète l'annexe G (instructions de reproductibilité).

> **Ce que « reproduire » veut dire ici.** On reproduit la **procédure** : mêmes données, même partition, même configuration, même protocole. Des chiffres identiques à la dernière décimale ne sont pas garantis, car le déterminisme strict du GPU n'était pas activé (annexe G.4). Attendez-vous à des résultats du même ordre, pas identiques.

## 0. Prérequis

| | Minimum | Utilisé dans le TFG |
|---|---|---|
| Python | 3.10 | 3.10.8 |
| TensorFlow | 2.15.1 | 2.15.1 |
| GPU | n'importe quel NVIDIA avec ≥ 8 Go (fonctionne sur CPU, mais prend des heures) | NVIDIA A30 24 Go, CUDA 12.2 |
| Disque | ~3 Go (ZIP + PNG) | |
| Système | Linux ou **WSL2** sous Windows (TensorFlow n'utilise plus le GPU sous Windows natif) | Linux (cluster HPC) |

```bash
git clone https://github.com/<votre-utilisateur>/detector-tumores-cnn.git
cd detector-tumores-cnn/training
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # sous Linux avec GPU : pip install "tensorflow[and-cuda]==2.15.1"
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

> **scikit-learn < 1.8.** La version 1.8 a changé la façon dont `StratifiedGroupKFold` mélange les données et produit des plis différents. Les versions 1.5.2, 1.6.1 et 1.7.2 reproduisent octet pour octet les plis archivés (vérifié).

---

## Voie A — Rapide : préparer les données et entraîner le modèle final (≈ 30 min avec GPU)

### A.1 Télécharger Cheng et construire le manifeste

```bash
python herramientas/construir_manifest_cheng.py \
    --root ./tfg_run \
    --split ../data/splits/split_multiclase_cheng.csv
```

Ce qu'il fait : télécharge les 4 ZIP depuis figshare (~880 Mo) en vérifiant le MD5, convertit les 3 064 fichiers `.mat` en PNG avec la fonction de prétraitement originale, attribue à chaque patient son sous-ensemble de la partition figée et **vérifie que l'empreinte de la partition est `fda7e2da…`**. Si quelque chose ne correspond pas, il s'arrête.

Sortie attendue à la fin :

```
  huella del split : fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092
OK: .../tfg_run/results/artifacts/manifest_final.csv  (3064 cortes, 233 pacientes, split verificado)
tumor_type  glioma  meningioma  pituitario
subset
test            12          10          13
train           63          62          38
val             14          10          11
```

Si figshare bloque le téléchargement automatique (certains réseaux renvoient 403), téléchargez les 4 ZIP à la main depuis la [page du jeu de données](https://doi.org/10.6084/m9.figshare.1512427), extrayez les `.mat` dans `tfg_run/data/raw/cheng_mat/` et lancez le script avec `--sin-descarga` (« sans téléchargement »).

### A.2 Entraîner le modèle final et l'évaluer une seule fois sur le test

```bash
python entrenar_modelo_final.py --root ./tfg_run
```

- Entraîne ResNet50‑FT sur les 198 patients d'entraînement+validation (6 époques de tête + 6 d'ajustement fin, graine 42).
- Enregistre `tfg_run/results/tumor_type/final/resnet50_ft_cheng_final.keras` et **`.h5`** (c'est ce dernier qu'utilise le backend Cloud Run).
- Évalue **une seule fois** les 35 patients de test et écrit `resultado_evaluacion_final.json`, les prédictions par coupe et par patient, et les fichiers `.lock`. Si vous le relancez dans le même dossier, il refuse d'ouvrir le test une seconde fois.

Référence du TFG : balanced accuracy 93,89 %, accuracy 94,29 %, macro-F1 93,98 %.

> Ce script est une **réimplémentation** du protocole C.1 : le notebook exact du job 70930 n'a pas été conservé. Il reprend ligne à ligne la recette de `robustez_5cv/ejecutar_fold_5cv_robustez.py`, qui est du code original.

---

## Voie B — Validation de robustesse à 5 plis avec les scripts originaux

Les scripts de `training/robustez_5cv/` sont **exactement** ceux exécutés sur le cluster (jobs 71101, 71102, 71107). Le chemin du cluster y est écrit en dur ; remplacez-le par le vôtre (l'empreinte du script changera, c'est normal) :

```bash
cd training/robustez_5cv
RUTA=$(realpath ../tfg_run)          # le dossier --root de la voie A
sed -i "s#/shared/home/FYK3492/TFG_tumores/tfg_run#$RUTA#" preparar_5cv_robustez.py ejecutar_fold_5cv_robustez.py agregar_5cv_robustez.py
# dossiers qui existaient déjà sur le cluster et que les scripts supposent présents
mkdir -p "$RUTA/frozen/robustez_5cv" "$RUTA/results/diagnostics/robustez_5cv"
# le protocole rédigé avant l'exécution (les scripts enregistrent son SHA-256 dans chaque résultat)
cp ../../docs/protocolos/C2_validacion_robustez_5cv.md "$RUTA/frozen/robustez_5cv/protocolo_5cv_robustez.md"

python preparar_5cv_robustez.py                    # 1) crée les 5 plis (StratifiedGroupKFold, graine 20260904)
for i in 0 1 2 3 4; do                             # 2) un modèle indépendant par pli (graines 1001..1005)
    python ejecutar_fold_5cv_robustez.py --fold-index $i
done
python agregar_5cv_robustez.py                     # 3) vérifie la couverture des 233 patients et agrège
```

Vérifier que vos plis sont ceux du TFG :

```bash
diff <(python -m json.tool $RUTA/results/tumor_type/robustez_5cv/splits/fold_1.json) \
     <(python -m json.tool ../../data/splits/robustez_5cv/fold_1.json) && echo "pli 1 identique"
```

Comparez votre `resultado_5cv_validacion_robustez.json` avec [`results/robustez_5cv/`](../../results/robustez_5cv) : moyenne de référence 89,44 % ± 4,64 pts.

---

## Voie C — Pipeline original complet (notebook)

`training/notebooks/02_pipeline_patient_level_multiclase.ipynb` est le pipeline complet tel qu'utilisé pour le benchmark multiclasse : il télécharge Cheng **et IXI**, construit le manifeste de 5 872 images, détecte les doublons, vérifie la partition figée, reproduit la référence historique et exécute le benchmark d'architectures. Ses commentaires sont en espagnol.

1. Définissez le dossier de travail : `export TFG_PROJECT_ROOT=$PWD/tfg_run_nb` (sur Colab, `/content/tfg_tumores` par défaut).
2. Copiez la partition figée : `mkdir -p $TFG_PROJECT_ROOT/frozen && cp ../data/splits/split_multiclase_cheng.* $TFG_PROJECT_ROOT/frozen/`
3. Choisissez quoi exécuter dans la **section 1.5** du notebook :
   - `RUN_MODE = "baseline_only"` → données + partition + référence CNN maison (contrôle de régression).
   - `RUN_MODE = "multiclass_benchmark"` → en plus, les 5 architectures avec GroupCV à 3 plis (tableau 4.2).
   - `RUN_MODE = "full"` et `RUN_BINARY_SECONDARY = True` → en plus, la tâche binaire historique avec IXI.
4. Exécutez tout : `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/02_pipeline_patient_level_multiclase.ipynb`

Avertissements :
- Le notebook **s'arrête** si la partition recalculée ne correspond pas à la partition figée : c'est voulu.
- Le téléchargement d'IXI peut échouer (le serveur de l'Imperial bloque certaines IP) ; le notebook réessaie et utilise Zenodo en secours.
- Le **contrôle de régression** de la référence peut arrêter l'exécution (c'est arrivé au job 70886 ; voir [04_resultats.md](04_resultats.md#41-référence-historique-cnn-maison-entraîné-de-zéro)).
- Les phases d'ajustement fin et d'ensemble (protocoles C.3 et C.4) **ne figurent pas dans cette version du notebook** (annexe A.3). Leurs protocoles sont dans [`protocolos/`](../protocolos) pour qui voudrait les réimplémenter.

`01_reconstruccion_dataset_y_control_3clases.ipynb` est la version antérieure de l'étape de reconstruction (job 70436), avec ses sorties ; elle est conservée comme référence historique. Voir [`training/notebooks/baseline_multiclase_cheng.md`](../../training/notebooks/baseline_multiclase_cheng.md) pour le lien entre les versions.

---

## Voie D — Sur un cluster SLURM

Les fichiers `.slurm` sont les originaux. Adaptez-les à votre cluster (partition, compte, modules et chemins) :

| Script | Ce qu'il lance | Ressources originales |
|---|---|---|
| `training/slurm/run_TFG_tumores_cerebrales_patient_level_multiclase.slurm` | notebook 02 via `nbconvert` | modèle avec `<PARTITION>`, `<ACCOUNT>`… |
| `training/robustez_5cv/run_preparar_5cv_robustez.slurm` | création des plis | 2 CPU, 4 Go, sans GPU |
| `training/robustez_5cv/run_5cv_robustez_array.slurm` | les 5 plis en *job array* (`--array=0-4%1`) | 1× A30, 8 CPU, 64 Go, 30 min |
| `training/robustez_5cv/run_agregar_5cv_robustez.slurm` | agrégation | 2 CPU, 4 Go |
| `training/robustez_5cv/run_export_final_5cv.slurm` | archivage des résultats en `tar` | sans GPU |

```bash
sbatch run_preparar_5cv_robustez.slurm
sbatch run_5cv_robustez_array.slurm           # attendez qu'il termine
sbatch run_agregar_5cv_robustez.slurm
```

---

## Bonnes pratiques à reprendre dans votre propre projet

1. **Découpez par patient** (`StratifiedGroupKFold(groups=patient_id)`), jamais par image.
2. **Figez la partition** dans un fichier avec son empreinte et faites en sorte que le code s'arrête si elle change.
3. **Rédigez le protocole d'évaluation avant d'ouvrir le test**, conservez-le avec son empreinte et n'ouvrez le test qu'une fois.
4. **Entraînez un classifieur de contrôle sans réseau de neurones.** S'il obtient un score élevé, soupçonnez un raccourci (scanner, hôpital, format).
5. **Évaluez par patient**, l'unité qui compte, et utilisez la balanced accuracy si les classes sont déséquilibrées.
6. **Rapportez la variabilité** (validation croisée) en plus du chiffre d'un test unique.
