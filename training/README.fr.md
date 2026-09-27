**Langue :** [Español](README.md) · [English](README.en.md) · **Français**

# training/

Code pour préparer les données, entraîner et évaluer. Le guide complet, avec les sorties attendues, se trouve dans [`docs/fr/05_guide_reproduction.md`](../docs/fr/05_guide_reproduction.md). Les commentaires du code et les messages de console sont en espagnol.

| Chemin | Ce que c'est | Origine |
|---|---|---|
| `requirements.txt` | Environnement (TensorFlow 2.15.1, scikit-learn < 1.8) | ajouté |
| `herramientas/construir_manifest_cheng.py` | Télécharge Cheng depuis figshare, génère les PNG avec le prétraitement original et le `manifest_final.csv` avec la partition figée (vérifie l'empreinte `fda7e2da…`) | ajouté |
| `entrenar_modelo_final.py` | Réimplémentation du protocole C.1 : entraîne ResNet50‑FT sur 198 patients et évalue une seule fois les 35 patients de test ; exporte `.keras` et `.h5` | ajouté |
| `notebooks/02_pipeline_patient_level_multiclase.ipynb` | Pipeline complet : téléchargement de Cheng + IXI, manifeste, doublons, partition figée, référence, benchmark de 5 architectures, tâche binaire historique | **original** |
| `notebooks/01_reconstruccion_dataset_y_control_3clases.ipynb` | Version de l'étape de reconstruction (job 70436), avec ses sorties | **original** |
| `notebooks/baseline_multiclase_cheng.md` | Document qui a figé la partition multiclasse et la référence de 79,7 % | **original** |
| `robustez_5cv/*.py`, `robustez_5cv/*.slurm` | Validation croisée à 5 plis telle qu'elle a été exécutée (jobs 71101, 71102, 71107) | **original** |
| `slurm/run_TFG_…_multiclase.slurm` | Lanceur SLURM du notebook 02 | **original** |

Les scripts originaux contiennent le chemin du cluster en dur (`/shared/home/…/tfg_run`). Pour les utiliser, remplacez-le par le vôtre (le guide indique la commande `sed`). Ils n'ont pas été modifiés ici, car chaque résultat enregistre le SHA‑256 du script qui l'a produit.

Démarrage rapide :

```bash
pip install -r requirements.txt
python herramientas/construir_manifest_cheng.py --root ./tfg_run --split ../data/splits/split_multiclase_cheng.csv
python entrenar_modelo_final.py --root ./tfg_run
```

Les deux scripts ajoutés ont été testés de bout en bout avec des données synthétiques imitant la structure de Cheng (même liste de patients et de coupes) : le manifeste reproduit l'empreinte de la partition, et l'entraînement, l'export `.h5` et l'évaluation fonctionnent. Ils n'ont pas pu être exécutés avec les vraies données et les vrais poids depuis l'environnement où le dépôt a été préparé.
