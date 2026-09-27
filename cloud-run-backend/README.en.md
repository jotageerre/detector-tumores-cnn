**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# cloud-run-backend — inference API (`run-model` service)

Flask API deployed on Google Cloud Run. It serves two models and a 3D viewer:

| Model (`"model"`) | File in `MODEL_BUCKET` | Output |
|---|---|---|
| `cheng` (final thesis model) | `resnet50_ft_cheng_final.h5` | glioma / meningioma / pituitary + Grad-CAM |
| `binary` (historical, default) | `brain_tumor_cnn.h5` | tumour / no tumour + Grad-CAM if score ≥ 0.9 |

NIfTI volumes (`.nii`, `.nii.gz`) are always processed with the binary model, slice by slice, and generate a 3D viewer (vtk.js) in HTML.

## Endpoints

- `GET /health` → status and loaded models.
- `POST /predict` → JSON `{"image_path" | "file_path": "gs://bucket/file", "model": "cheng" | "binary"}`. Returns `label`, `prediction`, `classes_probability`, `gradcam_url` and, for NIfTI, `render3d_url`, `best_slice`, `top10_slices`…

## Environment variables

| Variable | Default | Use |
|---|---|---|
| `MODEL_BUCKET` | `cnn-models-bucket` | Private bucket with the `.h5` files |
| `CHENG_MODEL_BLOB` | `resnet50_ft_cheng_final.h5` | |
| `BINARY_MODEL_BLOB` | `brain_tumor_cnn.h5` | |
| `CHENG_CLASSES` | `glioma,meningioma,pituitary` | Same order as the training softmax output |
| `CHENG_PREPROCESS` | `resnet` | ResNet50's `preprocess_input` |
| `BUCKET_GRADCAM_2D` | `mri-bucket-heatmap` | Public bucket for 2D results |
| `BUCKET_RENDER_3D` | `mri-bucket-3d` | Public bucket for the 3D viewer |
| `WEB_CONCURRENCY` | — | **Set it to `1`**: a single worker (see the 503 incident) |

There are no keys in the code: credentials come from the Cloud Run service account.

## Run locally

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
gcloud auth application-default login                # access to the buckets
export MODEL_BUCKET=… BUCKET_GRADCAM_2D=… BUCKET_RENDER_3D=…
python main.py                                       # http://localhost:8080/health
```

## Deploy

Full guide, from creating the project to the permissions: [`docs/en/06_google_cloud_deployment.md`](../docs/en/06_google_cloud_deployment.md). Summary:

```bash
gcloud run deploy run-model --source . --region us-central1 \
  --memory 8Gi --cpu 2 --concurrency 1 --timeout 900 \
  --set-env-vars MODEL_BUCKET=…,BUCKET_GRADCAM_2D=…,BUCKET_RENDER_3D=…,WEB_CONCURRENCY=1 \
  --allow-unauthenticated
```

The `Procfile` starts gunicorn with a single worker. With 4 GiB and several workers, each one loaded a copy of ResNet50 and the service returned 503 (Appendix J).

`requirements.txt` does not pin versions. If a `.h5` trained with TensorFlow 2.15 does not load on a newer version, pin `tensorflow==2.15.1` (and Python 3.10/3.11).
