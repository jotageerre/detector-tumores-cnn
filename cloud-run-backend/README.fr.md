**Langue :** [Español](README.md) · [English](README.en.md) · **Français**

# cloud-run-backend — API d'inférence (service `run-model`)

API Flask déployée sur Google Cloud Run. Elle sert deux modèles et une visionneuse 3D :

| Modèle (`"model"`) | Fichier dans `MODEL_BUCKET` | Sortie |
|---|---|---|
| `cheng` (modèle final du TFG) | `resnet50_ft_cheng_final.h5` | gliome / méningiome / hypophysaire + Grad-CAM |
| `binary` (historique, par défaut) | `brain_tumor_cnn.h5` | tumeur / pas de tumeur + Grad-CAM si score ≥ 0,9 |

Les volumes NIfTI (`.nii`, `.nii.gz`) sont toujours traités avec le modèle binaire, coupe par coupe, et génèrent une visionneuse 3D (vtk.js) en HTML.

## Points d'accès

- `GET /health` → état et modèles chargés.
- `POST /predict` → JSON `{"image_path" | "file_path": "gs://bucket/fichier", "model": "cheng" | "binary"}`. Renvoie `label`, `prediction`, `classes_probability`, `gradcam_url` et, pour NIfTI, `render3d_url`, `best_slice`, `top10_slices`…

## Variables d'environnement

| Variable | Par défaut | Usage |
|---|---|---|
| `MODEL_BUCKET` | `cnn-models-bucket` | Bucket privé contenant les `.h5` |
| `CHENG_MODEL_BLOB` | `resnet50_ft_cheng_final.h5` | |
| `BINARY_MODEL_BLOB` | `brain_tumor_cnn.h5` | |
| `CHENG_CLASSES` | `glioma,meningioma,pituitary` | Même ordre que la sortie softmax de l'entraînement |
| `CHENG_PREPROCESS` | `resnet` | `preprocess_input` de ResNet50 |
| `BUCKET_GRADCAM_2D` | `mri-bucket-heatmap` | Bucket public des résultats 2D |
| `BUCKET_RENDER_3D` | `mri-bucket-3d` | Bucket public de la visionneuse 3D |
| `WEB_CONCURRENCY` | — | **Mettez `1`** : un seul worker (voir l'incident 503) |

Aucune clé dans le code : les identifiants proviennent du compte de service de Cloud Run.

## Exécuter en local

```bash
python -m venv venv && source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
gcloud auth application-default login                # accès aux buckets
export MODEL_BUCKET=… BUCKET_GRADCAM_2D=… BUCKET_RENDER_3D=…
python main.py                                       # http://localhost:8080/health
```

## Déployer

Guide complet, de la création du projet aux permissions : [`docs/fr/06_deploiement_google_cloud.md`](../docs/fr/06_deploiement_google_cloud.md). Résumé :

```bash
gcloud run deploy run-model --source . --region us-central1 \
  --memory 8Gi --cpu 2 --concurrency 1 --timeout 900 \
  --set-env-vars MODEL_BUCKET=…,BUCKET_GRADCAM_2D=…,BUCKET_RENDER_3D=…,WEB_CONCURRENCY=1 \
  --allow-unauthenticated
```

Le `Procfile` démarre gunicorn avec un seul worker. Avec 4 Gio et plusieurs workers, chacun chargeait une copie de ResNet50 et le service renvoyait 503 (annexe J).

`requirements.txt` ne fixe pas les versions. Si un `.h5` entraîné avec TensorFlow 2.15 ne se charge pas avec une version plus récente, fixez `tensorflow==2.15.1` (et Python 3.10/3.11).
