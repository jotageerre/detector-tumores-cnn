**Language:** [Español](../06_despliegue_google_cloud.md) · **English** · [Français](../fr/06_deploiement_google_cloud.md)

# 6. Setting up your own Google Cloud Run service and the desktop app

This guide deploys, from scratch, the same architecture used in the thesis (chapter 5 and Appendix J): a desktop app that uploads the image to Cloud Storage and a Cloud Run API that classifies it.

<p align="center"><img src="../img/arquitectura_app.jpg" width="70%"></p>

```
Desktop app (Tkinter)
  1. uploads the image/NIfTI  ──►  GRADCAM_2D / RENDER_3D bucket  (Cloud Storage)
  2. POST /predict {"file_path": "gs://…", "model": "cheng"}  ──►  Cloud Run "run-model"
                                                                  ├─ downloads the .h5 model from MODEL_BUCKET (once)
                                                                  ├─ predicts + Grad-CAM (or 3D viewer for NIfTI)
                                                                  └─ uploads the result to the bucket and returns JSON with URLs
  3. shows the prediction, probabilities and Grad-CAM; opens the 3D viewer in the browser
```

> ⚠️ **Privacy.** As designed, the result buckets are **publicly readable** (Grad-CAM and 3D-viewer URLs open directly). Do not upload images of real patients. Use only public data such as Cheng's.

## 6.1 Requirements

- A Google Cloud account with billing enabled (Cloud Run has a free tier, but with 8 GiB of memory every request consumes resources).
- The [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and logged in: `gcloud auth login`.
- The models:
  - `resnet50_ft_cheng_final.h5` (multiclass, final model). Download it from this repository's **Releases** section or generate it with `training/entrenar_modelo_final.py`.
  - `brain_tumor_cnn.h5` (historical binary). Optional, but the backend uses it by default when no model is given and **always** for NIfTI volumes. Remember it comes from the binary task affected by source bias (see [01](01_project_history.md#stage-3--source-bias-and-why-the-binary-task-was-dropped)).

## 6.2 Variables (replace them with yours)

Bucket names are **unique across all of Google Cloud**: you cannot reuse the thesis ones.

```bash
export PROJECT_ID="my-tumour-project"
export REGION="us-central1"
export MODEL_BUCKET="${PROJECT_ID}-models"
export BUCKET_2D="${PROJECT_ID}-heatmap"
export BUCKET_3D="${PROJECT_ID}-render3d"
```

## 6.3 Project and APIs

```bash
gcloud projects create $PROJECT_ID            # or use an existing one
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com
```

## 6.4 Buckets

```bash
# Models: private
gcloud storage buckets create gs://$MODEL_BUCKET --location=$REGION --uniform-bucket-level-access

# Results: public read (the backend returns https://storage.googleapis.com/... URLs)
for B in $BUCKET_2D $BUCKET_3D; do
  gcloud storage buckets create gs://$B --location=$REGION --uniform-bucket-level-access
  gcloud storage buckets add-iam-policy-binding gs://$B --member=allUsers --role=roles/storage.objectViewer
done

# Recommended: automatically delete uploads after 1 day
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":1}}]}' > lifecycle.json
gcloud storage buckets update gs://$BUCKET_2D --lifecycle-file=lifecycle.json
gcloud storage buckets update gs://$BUCKET_3D --lifecycle-file=lifecycle.json
```

Upload the models:

```bash
gcloud storage cp resnet50_ft_cheng_final.h5 brain_tumor_cnn.h5 gs://$MODEL_BUCKET/
```

## 6.5 Backend service account

```bash
gcloud iam service-accounts create run-model-sa --display-name="Cloud Run run-model"
SA="run-model-sa@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding gs://$MODEL_BUCKET --member="serviceAccount:$SA" --role=roles/storage.objectViewer
for B in $BUCKET_2D $BUCKET_3D; do
  gcloud storage buckets add-iam-policy-binding gs://$B --member="serviceAccount:$SA" --role=roles/storage.objectAdmin
done
```

The backend has no key in its code: it uses this service account's identity.

## 6.6 Deploy the API

```bash
cd cloud-run-backend
gcloud run deploy run-model \
  --source . \
  --region $REGION \
  --service-account $SA \
  --memory 8Gi --cpu 2 \
  --concurrency 1 \
  --timeout 900 \
  --set-env-vars MODEL_BUCKET=$MODEL_BUCKET,BUCKET_GRADCAM_2D=$BUCKET_2D,BUCKET_RENDER_3D=$BUCKET_3D,WEB_CONCURRENCY=1 \
  --allow-unauthenticated
```

- `--source .` builds the image with Cloud Build from `requirements.txt` and the `Procfile` (gunicorn with **a single worker**).
- `--memory 8Gi`, `--concurrency 1` and `WEB_CONCURRENCY=1` are the fix for the production incident described below. Do not lower them.
- `--allow-unauthenticated` leaves the API open to anyone who knows the URL. For private use, remove it and call it with an identity token.

When it finishes, `gcloud` prints the service URL (`https://run-model-xxxxx.run.app`).

## 6.7 Test the API

```bash
URL=$(gcloud run services describe run-model --region $REGION --format='value(status.url)')
curl $URL/health

# upload a test image and request a prediction from the multiclass model
gcloud storage cp example.png gs://$BUCKET_2D/tests/example.png
curl -X POST $URL/predict -H "Content-Type: application/json" \
     -d "{\"image_path\": \"gs://$BUCKET_2D/tests/example.png\", \"model\": \"cheng\"}"
```

The first request takes longer (cold start: the ~170 MB model is downloaded and loaded).

## 6.8 Configure the desktop app

1. **Client service account** (can only upload files):

   ```bash
   gcloud iam service-accounts create scania-client
   CSA="scania-client@${PROJECT_ID}.iam.gserviceaccount.com"
   for B in $BUCKET_2D $BUCKET_3D; do
     gcloud storage buckets add-iam-policy-binding gs://$B --member="serviceAccount:$CSA" --role=roles/storage.objectCreator
   done
   gcloud iam service-accounts keys create desktop-client/gcp-service-account.json --iam-account=$CSA
   ```

   That JSON is a **private key**: it is in `.gitignore`, never upload it. If it leaks, revoke it under *IAM → Service accounts → Keys*.

2. **Buckets and URL**: set these environment variables before opening the app (on Windows, `set VAR=value` in the same console):

   ```bash
   export SCANIA_BUCKET_2D=$BUCKET_2D
   export SCANIA_BUCKET_3D=$BUCKET_3D
   export SCANIA_API_URL="$URL/predict"
   ```

3. The URL can also be changed inside the app (*Settings*) or by copying `config.example.json` to `config.json` and editing `api_url`; that value is kept between sessions.

4. Install and run (see [`desktop-client/README.en.md`](../../desktop-client/README.en.md)):

   ```bash
   cd desktop-client
   python -m venv myenv && myenv\Scripts\activate      # Windows
   pip install -r requirements.txt
   python main.py
   ```

<p align="center"><img src="../img/app_resultado.jpg" width="48%"> <img src="../img/app_visor3d.jpg" width="48%"></p>

## Known issue: 503 error

**Symptom:** the service returned 503 on incoming requests. **Cause:** with 4 GiB of memory, gunicorn started several workers and each one loaded its own copy of ResNet50; together they exhausted the container's memory. **Fix applied** (24/09/2026, revision `run-model-00062-p5z`, verified with `/health`):

```bash
gcloud run services update run-model --memory=8Gi --concurrency=1 --update-env-vars=WEB_CONCURRENCY=1
```

## Known difference between training and inference

During training, each slice was cropped to the brain bounding box and normalised by the 1st–99th percentiles before resizing ([02_data.md](02_data.md#23-preprocessing-of-each-slice-same-for-the-whole-project)). For standalone 2D images, the backend only resizes to 224×224 and applies `preprocess_input`. With images that are already cropped (such as Cheng's) the effect is small, but with other sources it may degrade the prediction. Applying the same `_a_png` in the backend is a pending improvement.

## Approximate costs

With `min-instances=0` (the default) you only pay while a request is being processed, but every cold start downloads the model again. For a demo, the cost usually stays within the free tier or a few cents; check the [pricing calculator](https://cloud.google.com/products/calculator) before leaving it open.
