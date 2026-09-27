**Idioma:** **Español** · [English](README.en.md) · [Français](README.fr.md)

# cloud-run-backend — API de inferencia (servicio `run-model`)

API Flask desplegada en Google Cloud Run. Sirve dos modelos y un visor 3D:

| Modelo (`"model"`) | Fichero en `MODEL_BUCKET` | Salida |
|---|---|---|
| `cheng` (modelo final del TFG) | `resnet50_ft_cheng_final.h5` | glioma / meningioma / pituitary + Grad-CAM |
| `binary` (histórico, por defecto) | `brain_tumor_cnn.h5` | tumor / no tumor + Grad-CAM si score ≥ 0,9 |

Los volúmenes NIfTI (`.nii`, `.nii.gz`) siempre se procesan con el modelo binario, corte a corte, y generan un visor 3D (vtk.js) en HTML.

## Endpoints

- `GET /health` → estado y modelos cargados.
- `POST /predict` → JSON `{"image_path" | "file_path": "gs://bucket/fichero", "model": "cheng" | "binary"}`. Devuelve `label`, `prediction`, `classes_probability`, `gradcam_url` y, para NIfTI, `render3d_url`, `best_slice`, `top10_slices`…

## Variables de entorno

| Variable | Por defecto | Uso |
|---|---|---|
| `MODEL_BUCKET` | `cnn-models-bucket` | Bucket privado con los `.h5` |
| `CHENG_MODEL_BLOB` | `resnet50_ft_cheng_final.h5` | |
| `BINARY_MODEL_BLOB` | `brain_tumor_cnn.h5` | |
| `CHENG_CLASSES` | `glioma,meningioma,pituitary` | Mismo orden que la salida softmax del entrenamiento |
| `CHENG_PREPROCESS` | `resnet` | `preprocess_input` de ResNet50 |
| `BUCKET_GRADCAM_2D` | `mri-bucket-heatmap` | Bucket público de resultados 2D |
| `BUCKET_RENDER_3D` | `mri-bucket-3d` | Bucket público del visor 3D |
| `WEB_CONCURRENCY` | — | **Pon `1`**: un solo worker (ver incidente 503) |

No hay claves en el código: las credenciales vienen de la cuenta de servicio de Cloud Run.

## Ejecutar en local

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
gcloud auth application-default login                # acceso a los buckets
export MODEL_BUCKET=… BUCKET_GRADCAM_2D=… BUCKET_RENDER_3D=…
python main.py                                       # http://localhost:8080/health
```

## Desplegar

Guía completa, desde crear el proyecto hasta los permisos: [`docs/06_despliegue_google_cloud.md`](../docs/06_despliegue_google_cloud.md). Resumen:

```bash
gcloud run deploy run-model --source . --region us-central1 \
  --memory 8Gi --cpu 2 --concurrency 1 --timeout 900 \
  --set-env-vars MODEL_BUCKET=…,BUCKET_GRADCAM_2D=…,BUCKET_RENDER_3D=…,WEB_CONCURRENCY=1 \
  --allow-unauthenticated
```

El `Procfile` arranca gunicorn con un solo worker. Con 4 GiB y varios workers, cada uno cargaba una copia del ResNet50 y el servicio devolvía 503 (Anexo J).

`requirements.txt` no fija versiones. Si un `.h5` entrenado con TensorFlow 2.15 no carga en una versión más nueva, fija `tensorflow==2.15.1` (y Python 3.10/3.11).
