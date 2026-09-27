# 6. Montar tu propio servicio en Google Cloud Run y la app de escritorio

Esta guía despliega desde cero la misma arquitectura que usa el TFG (capítulo 5 y Anexo J de la memoria): una app de escritorio que sube la imagen a Cloud Storage y una API en Cloud Run que la clasifica.

<p align="center"><img src="img/arquitectura_app.jpg" width="70%"></p>

```
App de escritorio (Tkinter)
  1. sube la imagen/NIfTI  ──►  bucket GRADCAM_2D / RENDER_3D  (Cloud Storage)
  2. POST /predict {"file_path": "gs://…", "model": "cheng"}  ──►  Cloud Run "run-model"
                                                                  ├─ descarga el modelo .h5 de MODEL_BUCKET (una vez)
                                                                  ├─ predice + Grad-CAM (o visor 3D si es NIfTI)
                                                                  └─ sube el resultado al bucket y devuelve JSON con URLs
  3. muestra predicción, probabilidades y Grad-CAM; abre el visor 3D en el navegador
```

> ⚠️ **Privacidad.** Tal como está diseñado, los buckets de resultados son de **lectura pública** (las URLs de Grad-CAM y del visor 3D se abren directamente). No subas imágenes de pacientes reales. Usa solo datos públicos como los de Cheng.

## 6.1 Requisitos

- Una cuenta de Google Cloud con facturación activada (Cloud Run tiene capa gratuita, pero con 8 GiB de memoria cada petición consume).
- [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install) instalado y con sesión iniciada: `gcloud auth login`.
- Los modelos:
  - `resnet50_ft_cheng_final.h5` (multiclase, modelo final). Descárgalo de la sección **Releases** de este repositorio o genéralo con `training/entrenar_modelo_final.py`.
  - `brain_tumor_cnn.h5` (binario histórico). Opcional, pero el backend lo usa por defecto si no se indica modelo y **siempre** para volúmenes NIfTI. Recuerda que viene de la tarea binaria con sesgo de procedencia (ver [01](01_historia_del_proyecto.md#etapa-3--el-sesgo-de-procedencia-y-por-qué-se-abandonó-la-tarea-binaria)).

## 6.2 Variables (cámbialas por las tuyas)

Los nombres de bucket son **únicos en todo Google Cloud**: no puedes reutilizar los del TFG.

```bash
export PROJECT_ID="mi-proyecto-tumores"
export REGION="us-central1"
export MODEL_BUCKET="${PROJECT_ID}-modelos"
export BUCKET_2D="${PROJECT_ID}-heatmap"
export BUCKET_3D="${PROJECT_ID}-render3d"
```

## 6.3 Proyecto y APIs

```bash
gcloud projects create $PROJECT_ID            # o usa uno existente
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com storage.googleapis.com iam.googleapis.com
```

## 6.4 Buckets

```bash
# Modelos: privado
gcloud storage buckets create gs://$MODEL_BUCKET --location=$REGION --uniform-bucket-level-access

# Resultados: lectura pública (el backend devuelve URLs https://storage.googleapis.com/...)
for B in $BUCKET_2D $BUCKET_3D; do
  gcloud storage buckets create gs://$B --location=$REGION --uniform-bucket-level-access
  gcloud storage buckets add-iam-policy-binding gs://$B --member=allUsers --role=roles/storage.objectViewer
done

# Recomendado: borrar automáticamente lo subido tras 1 día
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":1}}]}' > lifecycle.json
gcloud storage buckets update gs://$BUCKET_2D --lifecycle-file=lifecycle.json
gcloud storage buckets update gs://$BUCKET_3D --lifecycle-file=lifecycle.json
```

Sube los modelos:

```bash
gcloud storage cp resnet50_ft_cheng_final.h5 brain_tumor_cnn.h5 gs://$MODEL_BUCKET/
```

## 6.5 Cuenta de servicio del backend

```bash
gcloud iam service-accounts create run-model-sa --display-name="Cloud Run run-model"
SA="run-model-sa@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding gs://$MODEL_BUCKET --member="serviceAccount:$SA" --role=roles/storage.objectViewer
for B in $BUCKET_2D $BUCKET_3D; do
  gcloud storage buckets add-iam-policy-binding gs://$B --member="serviceAccount:$SA" --role=roles/storage.objectAdmin
done
```

El backend no lleva ninguna clave en el código: usa la identidad de esta cuenta de servicio.

## 6.6 Desplegar la API

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

- `--source .` construye la imagen con Cloud Build a partir de `requirements.txt` y del `Procfile` (gunicorn con **un solo worker**).
- `--memory 8Gi`, `--concurrency 1` y `WEB_CONCURRENCY=1` son la corrección del incidente de producción descrito abajo. No los bajes.
- `--allow-unauthenticated` deja la API abierta a quien conozca la URL. Para uso privado, quítalo y llama con un token de identidad.

Al terminar, `gcloud` imprime la URL del servicio (`https://run-model-xxxxx.run.app`).

## 6.7 Probar la API

```bash
URL=$(gcloud run services describe run-model --region $REGION --format='value(status.url)')
curl $URL/health

# sube una imagen de prueba y pide la predicción del modelo multiclase
gcloud storage cp ejemplo.png gs://$BUCKET_2D/pruebas/ejemplo.png
curl -X POST $URL/predict -H "Content-Type: application/json" \
     -d "{\"image_path\": \"gs://$BUCKET_2D/pruebas/ejemplo.png\", \"model\": \"cheng\"}"
```

La primera petición tarda más (arranque en frío: descarga y carga del modelo de ~170 MB).

## 6.8 Configurar la app de escritorio

1. **Cuenta de servicio del cliente** (solo puede subir ficheros):

   ```bash
   gcloud iam service-accounts create scania-client
   CSA="scania-client@${PROJECT_ID}.iam.gserviceaccount.com"
   for B in $BUCKET_2D $BUCKET_3D; do
     gcloud storage buckets add-iam-policy-binding gs://$B --member="serviceAccount:$CSA" --role=roles/storage.objectCreator
   done
   gcloud iam service-accounts keys create desktop-client/gcp-service-account.json --iam-account=$CSA
   ```

   Ese JSON es una **clave privada**: está en `.gitignore`, nunca lo subas. Si se filtra, revócala en *IAM → Cuentas de servicio → Claves*.

2. **Buckets y URL**: define estas variables de entorno antes de abrir la app (en Windows, `set VAR=valor` en la misma consola):

   ```bash
   export SCANIA_BUCKET_2D=$BUCKET_2D
   export SCANIA_BUCKET_3D=$BUCKET_3D
   export SCANIA_API_URL="$URL/predict"
   ```

3. La URL también se puede cambiar dentro de la app (*Settings*) o copiando `config.example.json` a `config.json` y editando `api_url`; ese valor se guarda entre sesiones.

4. Instala y ejecuta (ver [`desktop-client/README.md`](../desktop-client/README.md)):

   ```bash
   cd desktop-client
   python -m venv myenv && myenv\Scripts\activate      # Windows
   pip install -r requirements.txt
   python main.py
   ```

<p align="center"><img src="img/app_resultado.jpg" width="48%"> <img src="img/app_visor3d.jpg" width="48%"></p>

## Problema conocido: error 503

**Síntoma:** el servicio devolvía 503 al recibir peticiones. **Causa:** con 4 GiB de memoria, gunicorn arrancaba varios workers y cada uno cargaba su propia copia del ResNet50; la suma agotaba la memoria del contenedor. **Solución aplicada** (24/09/2026, revisión `run-model-00062-p5z`, verificada con `/health`):

```bash
gcloud run services update run-model --memory=8Gi --concurrency=1 --update-env-vars=WEB_CONCURRENCY=1
```

## Diferencia conocida entre entrenamiento e inferencia

En el entrenamiento cada corte se recortaba a la caja del cerebro y se normalizaba por percentiles 1–99 antes de redimensionar ([02_datos.md](02_datos.md#23-preprocesado-de-cada-corte-igual-para-todo-el-proyecto)). El backend, para imágenes 2D sueltas, solo redimensiona a 224×224 y aplica `preprocess_input`. Con imágenes que ya vienen recortadas (como las de Cheng) el efecto es pequeño, pero con otras fuentes puede empeorar la predicción. Aplicar el mismo `_a_png` en el backend es una mejora pendiente.

## Costes orientativos

Con `min-instances=0` (por defecto) solo pagas mientras se procesa una petición, pero cada arranque en frío vuelve a descargar el modelo. Para una demo, el coste suele quedar dentro de la capa gratuita o en céntimos; revisa la [calculadora de precios](https://cloud.google.com/products/calculator) antes de dejarlo abierto.
