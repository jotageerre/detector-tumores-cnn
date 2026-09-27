# ============================================================
# main.py — ScanIA backend (Cloud Run / Cloud Functions gen2)
# ------------------------------------------------------------
# - Imagen (.jpg/.png/...) -> predicción + Grad-CAM 2D (si score>=0.9 en binario,
#   siempre en Cheng) usando uno de DOS modelos seleccionables:
#     "binary" -> brain_tumor_cnn.h5      (clasificación binaria: tumor/no_tumor)
#     "cheng"  -> resnet50_ft_cheng_final.h5 (clasificación multicategoría, dataset Cheng)
# - NIfTI (.nii / .nii.gz) -> scan por slices + pseudo-Grad heat 3D
#   + visor volumétrico premium (vtk.js) en HTML
#   (el pipeline NIfTI SIEMPRE usa el modelo binario, independientemente del
#    modelo seleccionado en la app, ya que produce un análisis volumétrico
#    binario por diseño)
#
# COMPATIBILIDAD (IMPORTANTE):
# - Se conservan TODAS las claves JSON que ya usa la app local:
#     type, label, prediction, gradcam_url,
#     nifti_filename, volume_shape, render3d_url,
#     best_slice, best_score, score_mean, top10_slices
# - Se AÑADEN claves nuevas (opcionales) sin romper nada:
#     model, threshold, tumor_probability, no_tumor_probability, slice_count,
#     slice_scores, top_slices, score_statistics, voxel_spacing,
#     model_version, request_id, processing_times, viewer_3d_url, warnings,
#     classes_probability (solo modelo "cheng")
# - Petición /predict: acepta un campo opcional "model" ("binary" | "cheng").
#   Si no viene, se usa "binary" por defecto (compatibilidad total con
#   peticiones antiguas que no incluían este campo).
# - upload_to_gcs NO usa make_public (UBLA). URL directa (bucket público por IAM).
# ============================================================

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from google.cloud import storage
from PIL import Image
from tensorflow.keras.preprocessing import image as kimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

# ==========================
# CONFIG (AJUSTA A TUS BUCKETS)  — se pueden sobreescribir por variables de entorno
# ==========================
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "cnn-models-bucket")

BUCKET_GRADCAM_2D = os.environ.get("BUCKET_GRADCAM_2D", "mri-bucket-heatmap")
BUCKET_RENDER_3D = os.environ.get("BUCKET_RENDER_3D", "mri-bucket-3d")

# ---- Modelo 1: "Binary Tumor Detection" (clasificación binaria) ----
# MODEL_BLOB / LAST_CONV_LAYER / MODEL_VERSION se mantienen como nombres de
# variable de entorno por compatibilidad con despliegues existentes.
BINARY_MODEL_BLOB = os.environ.get("BINARY_MODEL_BLOB", os.environ.get("MODEL_BLOB", "brain_tumor_cnn.h5"))
MODEL_INPUT_SIZE = (150, 150)  # tamaño de entrada del modelo binario (histórico)
BINARY_INPUT_SIZE = MODEL_INPUT_SIZE
BINARY_LAST_CONV_LAYER = os.environ.get("BINARY_LAST_CONV_LAYER", os.environ.get("LAST_CONV_LAYER", "conv2d_3"))
BINARY_MODEL_VERSION = os.environ.get("BINARY_MODEL_VERSION", os.environ.get("MODEL_VERSION", "brain_tumor_cnn_v1"))

# ---- Modelo 2: "Cheng Brain Tumor Classification" (multicategoría, dataset Cheng) ----
CHENG_MODEL_BLOB = os.environ.get("CHENG_MODEL_BLOB", "resnet50_ft_cheng_final.h5")
_cheng_input_size_raw = os.environ.get("CHENG_INPUT_SIZE", "224,224")
try:
    _cheng_w, _cheng_h = [int(v.strip()) for v in _cheng_input_size_raw.split(",")[:2]]
    CHENG_INPUT_SIZE = (_cheng_w, _cheng_h)
except Exception:
    CHENG_INPUT_SIZE = (224, 224)
CHENG_LAST_CONV_LAYER = os.environ.get("CHENG_LAST_CONV_LAYER", "")  # vacío => autodetección
CHENG_MODEL_VERSION = os.environ.get("CHENG_MODEL_VERSION", "resnet50_ft_cheng_v1")
# Cómo se preprocesa la imagen antes de entrar al modelo Cheng:
#   "resnet"  -> tf.keras.applications.resnet50.preprocess_input (típico en ResNet50 preentrenado)
#   "rescale" -> división simple por 255.0 (igual que el modelo binario)
CHENG_PREPROCESS = os.environ.get("CHENG_PREPROCESS", "resnet").strip().lower()

# IMPORTANTE — clases del modelo Cheng:
# No se inventan las clases. Por defecto se usan las 3 clases estándar del
# dataset público de Cheng (figshare, 2017): glioma, meningioma, pituitary.
# Si tu modelo fue entrenado con clases distintas o en otro orden, AJUSTA
# la variable de entorno CHENG_CLASSES (lista separada por comas, en el
# MISMO ORDEN que la salida softmax del modelo), por ejemplo:
#   CHENG_CLASSES="glioma,meningioma,pituitary,notumor"
CHENG_CLASSES = [c.strip() for c in os.environ.get(
    "CHENG_CLASSES", "glioma,meningioma,pituitary"
).split(",") if c.strip()]

# Tipos de modelo soportados por /predict
SUPPORTED_MODEL_TYPES = ("binary", "cheng")
DEFAULT_MODEL_TYPE = "binary"

# Alias de compatibilidad (código/scripts previos podían referenciar MODEL_VERSION
# a secas, asumiendo un único modelo binario).
MODEL_VERSION = BINARY_MODEL_VERSION

# NIfTI pipeline
THRESH_SCORE = float(os.environ.get("THRESH_SCORE", "0.75"))
CLASSIFY_THRESHOLD = float(os.environ.get("CLASSIFY_THRESHOLD", "0.5"))
STEP_3D = int(os.environ.get("STEP_3D", "2"))
MAX_SLICES_GRADCAM = int(os.environ.get("MAX_SLICES_GRADCAM", "60"))
HEAT_SHOW = float(os.environ.get("HEAT_SHOW", "0.45"))

# Tope de resolución del volumen que se manda al visor 3D.
# Un volumen con lado > VIEWER_MAX_DIM se submuestrea más para que el
# renderizado en el navegador sea fluido.
VIEWER_MAX_DIM = int(os.environ.get("VIEWER_MAX_DIM", "100"))

# Slice “uniforme” (no se tiene en cuenta)
UNIFORM_RANGE_EPS = 1e-6

# Límites de robustez
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "400"))
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
ALLOWED_NIFTI_EXT = (".nii", ".nii.gz")

# El HTML/CSS/JS del visor 3D va INCRUSTADO al final de este mismo fichero
# (constantes _VIEWER_HTML, _VIEWER_CSS, _VIEWER_JS). Así main.py es autónomo:
# no necesita carpetas templates/ ni static/ para desplegarse.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scania")

app = Flask(__name__)

# ==========================
# GLOBAL CACHE (carga única de cada modelo por instancia)
# ==========================
# _models_cache: model_type -> (model, grad_model, conv_layer_name_usada)
_models_cache: Dict[str, Tuple[Any, Any, str]] = {}

# Alias de compatibilidad: algunas partes del código (y despliegues previos)
# podían referenciar _model / _grad_model directamente. Se mantienen apuntando
# siempre al modelo binario una vez cargado.
_model = None
_grad_model = None


def _log(request_id: str, msg: str, **kw: Any) -> None:
    payload = {"request_id": request_id, "msg": msg}
    payload.update(kw)
    logger.info(json.dumps(payload, default=str))


# ==========================
# GCS HELPERS
# ==========================
def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError("La ruta debe empezar por gs://")
    path = gs_uri[5:]
    if "/" not in path:
        raise ValueError("gs:// URI inválida (falta el nombre del objeto)")
    bucket, blob = path.split("/", 1)
    return bucket, blob


def _safe_basename(name: str) -> str:
    """Evita path traversal: nos quedamos solo con el nombre de archivo."""
    return os.path.basename(name.replace("\\", "/")).strip() or "file"


def download_from_gcs(gs_uri: str, local_path: str) -> str:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"El objeto no existe: {gs_uri}")
    blob.reload()
    if blob.size and blob.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Archivo demasiado grande (> {MAX_UPLOAD_MB} MB)")
    blob.download_to_filename(local_path)
    return local_path


def upload_to_gcs(bucket_name: str, local_path: str, dest_name: str,
                  content_type: Optional[str] = None) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(local_path, content_type=content_type)
    # UBLA => NO ACLs => NO make_public()
    return f"https://storage.googleapis.com/{bucket_name}/{dest_name}"


# ==========================
# MODEL LOADING (ONCE POR TIPO DE MODELO)
# ==========================
def _resolve_last_conv_layer(model, configured_name: Optional[str] = None):
    """
    Localiza la capa (tensor) que se usará como última capa convolucional
    para Grad-CAM. Devuelve el objeto Layer de Keras (no el nombre).

    Estrategia:
      1) Si se da un nombre configurado, se busca esa capa directamente en
         el modelo y, si no está ahí, dentro de cualquier submodelo anidado
         (habitual cuando una red base tipo ResNet50 se incluye como una
         única capa, p.ej. `ResNet50(include_top=False)(inputs)`).
      2) Si no hay nombre configurado o no se encuentra, se autodetecta la
         última capa convolucional recorriendo el modelo (y submodelos
         anidados) de atrás hacia adelante.
    """
    if configured_name:
        try:
            return model.get_layer(configured_name)
        except ValueError:
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    try:
                        return layer.get_layer(configured_name)
                    except ValueError:
                        continue
            logger.warning(
                "No se encontró la capa configurada '%s'; se intentará autodetección.",
                configured_name,
            )

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if "conv" in sub_layer.__class__.__name__.lower():
                    return sub_layer
        elif "conv" in layer.__class__.__name__.lower():
            return layer

    return None


def build_grad_model(model, configured_layer_name: Optional[str] = None,
                     input_size: Optional[Tuple[int, int]] = None):
    """
    Construye el modelo auxiliar de Grad-CAM.

    Para modelos `Sequential` "planos" (como el binario) se reconstruye el
    grafo hacia adelante EXPLÍCITAMENTE sobre un `Input` nuevo, llamando a
    cada capa en orden y capturando su tensor de salida directamente. Esto
    evita depender de `layer.output` / `model.output`, que en Keras 3 pueden
    lanzar "The layer ... has never been called and thus has no defined
    output." para un `Sequential` recién cargado desde un .h5 legacy —
    incluso después de invocar el modelo una vez (el "calentamiento" de
    load_model_cached no basta: los objetos Layer que devuelve
    model.get_layer()/model.layers no siempre quedan con nodos válidos tras
    esa llamada). Reconstruir el grafo así es 100% fiable porque los nodos
    se crean en el momento, sobre el nuevo Input.

    Para el resto de arquitecturas (Functional / con submodelos anidados,
    como el ResNet50 del modelo Cheng, que ya funciona correctamente) se
    mantiene el enfoque original basado en `layer.output`.
    """
    if isinstance(model, tf.keras.Sequential):
        shape = None
        try:
            if model.input_shape:
                shape = tuple(model.input_shape[1:])
        except Exception:
            shape = None
        if not shape or any(d is None for d in shape):
            if input_size:
                shape = tuple(input_size) + (3,)
            else:
                shape = None
        if not shape:
            raise ValueError(
                "No se pudo determinar la forma de entrada del modelo Sequential "
                "para reconstruir el grafo de Grad-CAM."
            )

        inputs = tf.keras.Input(shape=shape)
        x = inputs
        layer_outputs: Dict[str, Any] = {}
        last_conv_name = None
        for layer in model.layers:
            x = layer(x)
            layer_outputs[layer.name] = x
            if "conv" in layer.__class__.__name__.lower():
                last_conv_name = layer.name

        target_name = configured_layer_name if configured_layer_name in layer_outputs else last_conv_name
        if not target_name or target_name not in layer_outputs:
            raise ValueError(
                "No se pudo determinar automáticamente la última capa "
                "convolucional del modelo Sequential."
            )
        grad_model = tf.keras.Model(inputs=inputs, outputs=[layer_outputs[target_name], x])
        return grad_model, target_name

    target_layer = _resolve_last_conv_layer(model, configured_layer_name)
    if target_layer is None:
        raise ValueError(
            "No se pudo determinar automáticamente la última capa convolucional. "
            "Define BINARY_LAST_CONV_LAYER o CHENG_LAST_CONV_LAYER con el nombre "
            "exacto de la capa (consulta model.summary())."
        )
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    return grad_model, target_layer.name


def _model_type_settings(model_type: str):
    if model_type == "binary":
        return {
            "blob": BINARY_MODEL_BLOB,
            "last_conv_layer": BINARY_LAST_CONV_LAYER,
            "version": BINARY_MODEL_VERSION,
            "input_size": BINARY_INPUT_SIZE,
        }
    if model_type == "cheng":
        return {
            "blob": CHENG_MODEL_BLOB,
            "last_conv_layer": CHENG_LAST_CONV_LAYER,
            "version": CHENG_MODEL_VERSION,
            "input_size": CHENG_INPUT_SIZE,
        }
    raise ValueError(f"model_type desconocido: {model_type}")


# Un lock por tipo de modelo evita que dos peticiones concurrentes en la
# misma instancia de Cloud Run intenten descargar/cargar el MISMO modelo a
# la vez la primera vez (causa de fallos intermitentes tipo "No se pudo
# cargar el modelo": dos hilos escribiendo/leyendo el mismo fichero .h5).
_model_load_locks: Dict[str, threading.Lock] = {mt: threading.Lock() for mt in SUPPORTED_MODEL_TYPES}


def load_model_cached(model_type: str = DEFAULT_MODEL_TYPE):
    """
    Carga (una sola vez por instancia de Cloud Run) el modelo indicado por
    `model_type` ("binary" o "cheng") y construye su grad_model asociado.
    Cada modelo se cachea de forma independiente en `_models_cache`.

    Thread-safe: usa un lock por model_type + un nombre de fichero temporal
    único por descarga, para que peticiones concurrentes que necesiten el
    mismo modelo por primera vez no colisionen entre sí.
    """
    global _model, _grad_model

    model_type = (model_type or DEFAULT_MODEL_TYPE).strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"model_type no soportado: {model_type}")

    if model_type in _models_cache:
        return _models_cache[model_type]

    lock = _model_load_locks[model_type]
    with lock:
        # Doble comprobación: otro hilo pudo haber terminado de cargarlo
        # mientras esperábamos a adquirir el lock.
        if model_type in _models_cache:
            return _models_cache[model_type]

        settings = _model_type_settings(model_type)
        local_path = f"/tmp/model_{model_type}_{uuid.uuid4().hex[:8]}.h5"

        try:
            client = storage.Client()
            bucket = client.bucket(MODEL_BUCKET)
            blob = bucket.blob(settings["blob"])
            blob.download_to_filename(local_path)

            model = tf.keras.models.load_model(local_path)

            # "Calentamos" el modelo con un forward pass dummy (defensivo; no
            # imprescindible para el camino Sequential de build_grad_model,
            # que ya no depende de esto, pero es barato y no hace daño para
            # el resto de arquitecturas). Si falla, lo registramos con el
            # traceback completo en vez de tragarnos la excepción en
            # silencio como antes.
            try:
                dummy_shape = (1,) + tuple(settings["input_size"]) + (3,)
                model(tf.zeros(dummy_shape, dtype=tf.float32))
            except Exception:
                logger.warning(
                    "No se pudo 'calentar' el modelo '%s' con un input dummy de "
                    "forma %s.", model_type, dummy_shape, exc_info=True,
                )

            grad_model, conv_layer_used = build_grad_model(
                model, settings["last_conv_layer"], settings.get("input_size")
            )
        finally:
            try:
                os.remove(local_path)
            except OSError:
                pass

        logger.info(
            "Modelo '%s' (%s) cargado. Capa conv usada para Grad-CAM: %s",
            model_type, settings["blob"], conv_layer_used,
        )

        _models_cache[model_type] = (model, grad_model, conv_layer_used)

        # Mantener los alias de compatibilidad apuntando al modelo binario.
        if model_type == "binary":
            _model, _grad_model = model, grad_model

        return _models_cache[model_type]


# ==========================
# IMAGE PIPELINE (2D)
# ==========================
def _preprocess_for_model(img_pil: Image.Image, input_size: Tuple[int, int], preprocess: str = "rescale") -> np.ndarray:
    """Redimensiona y normaliza una imagen PIL según el tipo de preprocesado del modelo."""
    img_resized = img_pil.resize(input_size)
    arr = tf.keras.preprocessing.image.img_to_array(img_resized)
    arr_batch = np.expand_dims(arr, axis=0)
    if preprocess == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input as _resnet_preprocess
        return _resnet_preprocess(arr_batch.astype(np.float32))
    # "rescale" (comportamiento histórico del modelo binario) u otro valor no reconocido
    return arr_batch.astype(np.float32) / 255.0


def generate_gradcam(img_pil: Image.Image, grad_model, input_size: Tuple[int, int],
                     class_index: int = 0, preprocess: str = "rescale") -> str:
    """
    Genera una imagen Grad-CAM superpuesta sobre `img_pil` para la clase
    `class_index` de `grad_model`. Sirve tanto para el modelo binario
    (class_index=0, salida sigmoide) como para el modelo Cheng
    (class_index=clase predicha, salida softmax multicategoría).
    """
    img_input = _preprocess_for_model(img_pil, input_size, preprocess)
    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_tensor)
        # En Keras 3, si `model.output` del modelo base ya era una lista (p.ej.
        # ocurre con el modelo Cheng), grad_model devuelve ese tensor envuelto
        # en una lista de un elemento en vez de un tensor "pelado", y
        # preds[:, class_index] fallaba con "list indices must be integers or
        # slices, not tuple" (visto en los logs: rompia el Grad-CAM del
        # modelo Cheng, aunque la clasificacion en si funcionaba).
        if isinstance(conv, (list, tuple)):
            conv = conv[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv0 = conv[0]
    heatmap = tf.reduce_sum(conv0 * pooled, axis=-1).numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_pil.size)
    colormap = cm.get_cmap("jet")
    heat = colormap(np.array(heatmap_img))[:, :, :3]
    heat = Image.fromarray((heat * 255).astype(np.uint8))

    final = Image.blend(img_pil.convert("RGB"), heat, alpha=0.4)
    out_path = f"/tmp/gradcam_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    final.save(out_path)
    return out_path


def generate_gradcam_2d(img_pil: Image.Image, grad_model) -> str:
    """Alias de compatibilidad: Grad-CAM del modelo binario (clase 0, /255.0)."""
    return generate_gradcam(img_pil, grad_model, BINARY_INPUT_SIZE, class_index=0, preprocess="rescale")


# ==========================
# VISOR 3D (vtk.js) — construcción del HTML
# ==========================
def _to_uint8_volume(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a).astype(np.float32)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def _b64_fortran(a_uint8: np.ndarray) -> str:
    # vtkImageData espera x más rápido -> flatten en orden Fortran
    return base64.b64encode(a_uint8.flatten(order="F").tobytes()).decode("ascii")


def build_brain_viewer_html(t1n_ds: np.ndarray,
                            heat_ds: np.ndarray,
                            spacing: Tuple[float, float, float],
                            meta: Dict[str, Any]) -> str:
    """
    Genera un HTML autónomo con el visor volumétrico vtk.js.
    Inyecta el CSS y el JS (de /static) y el volumen (base64 uint8) para que
    el fichero sea único y robusto (sin fetch externo salvo el CDN de vtk.js).
    """
    nx, ny, nz = t1n_ds.shape
    has_heat = bool(np.asarray(heat_ds).max() > 0)

    volume_payload = {
        "dims": [int(nx), int(ny), int(nz)],
        "spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "t1": _b64_fortran(_to_uint8_volume(t1n_ds)),
        "heat": _b64_fortran(_to_uint8_volume(heat_ds)) if has_heat else "",
        "hasHeat": has_heat,
        "heatShow": float(HEAT_SHOW),
        "meta": meta,
    }

    # Assets incrustados (constantes al final del fichero) -> main.py autónomo
    volume_json = json.dumps(volume_payload)
    html = (
        _VIEWER_HTML
        .replace("__CSS__", _VIEWER_CSS)
        .replace("__JS__", _VIEWER_JS)
        .replace("__VOLUME__", volume_json)
        .replace("__META__", json.dumps(meta))
    )
    return html


# ==========================
# NIFTI PIPELINE (3D)
# ==========================
def _normalize_slice_for_model(x: np.ndarray) -> Optional[np.ndarray]:
    x = np.nan_to_num(x).astype(np.float32)
    p1, p99 = np.percentile(x, (1, 99))
    if p99 - p1 < 1e-8:
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < UNIFORM_RANGE_EPS:
            return None
        return (x - mn) / (mx - mn + 1e-8)
    x = np.clip(x, p1, p99)
    return (x - p1) / (p99 - p1 + 1e-8)


def _cheng_classify_candidate_slices(vol: np.ndarray, cand_idx: np.ndarray,
                                     cheng_model, warnings: List[str]) -> Optional[Dict[str, Any]]:
    """
    Clasificación adicional del volumen con el modelo Cheng (multicategoría),
    SOLO para NIfTI y SOLO cuando el usuario tiene "Cheng" seleccionado.

    No se reclasifica todo el volumen: se reutilizan los mismos cortes que
    el modelo binario ya marcó como sospechosos de tumor (`cand_idx`, los
    mismos que se usan para el Grad-CAM), porque Cheng no tiene clase
    "sin tumor" y pasarle cortes sanos solo añadiría ruido. Se promedian
    (media, no voto mayoritario) las distribuciones de probabilidad por
    clase de cada corte candidato; el tipo más probable es el argmax de esa
    media. Al reutilizar los cortes ya seleccionados, no se dispara el
    tiempo de análisis.
    """
    if cheng_model is None or len(cand_idx) == 0:
        return None

    probs_sum = None
    used = 0
    for i in cand_idx:
        s = _normalize_slice_for_model(vol[:, :, i])
        if s is None:
            continue
        img = Image.fromarray((s * 255).astype(np.uint8)).convert("RGB").resize(CHENG_INPUT_SIZE)
        img_input = _preprocess_for_model(img, CHENG_INPUT_SIZE, CHENG_PREPROCESS)
        preds = cheng_model.predict(img_input, verbose=0)[0]
        probs_sum = preds if probs_sum is None else probs_sum + preds
        used += 1

    if probs_sum is None or used == 0:
        warnings.append("No se pudo clasificar ningún corte con el modelo Cheng.")
        return None

    mean_probs = probs_sum / used
    num_classes = int(mean_probs.shape[0])
    if len(CHENG_CLASSES) == num_classes:
        class_names = CHENG_CLASSES
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    best_idx = int(np.argmax(mean_probs))
    return {
        "cheng_label": class_names[best_idx],
        "cheng_probability": round(float(mean_probs[best_idx]), 6),
        "cheng_classes_probability": {
            class_names[i]: round(float(mean_probs[i]), 6) for i in range(num_classes)
        },
        "cheng_slices_used": used,
    }


def process_nifti(gs_uri: str, model, grad_model, request_id: str,
                  cheng_model=None) -> Dict[str, Any]:
    import nibabel as nib

    times: Dict[str, float] = {}
    t_all = time.time()
    warnings: List[str] = []

    # ---- Descarga ----
    t0 = time.time()
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    filename = _safe_basename(blob_name)  # respeta .nii o .nii.gz
    local_path = f"/tmp/{uuid.uuid4().hex}_{filename}"
    download_from_gcs(gs_uri, local_path)
    times["download_seconds"] = round(time.time() - t0, 3)

    # ---- Carga / validación ----
    t0 = time.time()
    try:
        nii = nib.load(local_path)
        vol = nii.get_fdata()
        zooms = nii.header.get_zooms()[:3]
    except Exception as e:
        raise ValueError(f"NIfTI ilegible o corrupto: {e}")

    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba un volumen 3D, se recibió ndim={vol.ndim}")

    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if vol.size == 0 or float(vol.max()) - float(vol.min()) < 1e-8:
        raise ValueError("El volumen está vacío o sin contraste.")

    H, W, D = vol.shape
    spacing_full = (
        float(zooms[0]) if len(zooms) > 0 and zooms[0] > 0 else 1.0,
        float(zooms[1]) if len(zooms) > 1 and zooms[1] > 0 else 1.0,
        float(zooms[2]) if len(zooms) > 2 and zooms[2] > 0 else 1.0,
    )

    scores = np.full(D, np.nan, dtype=np.float32)
    valid = np.zeros(D, dtype=bool)

    # 1) Score por slice (ignorando slices uniformes)
    for i in range(D):
        s = _normalize_slice_for_model(vol[:, :, i])
        if s is None:
            continue
        valid[i] = True
        img = Image.fromarray((s * 255).astype(np.uint8)).resize(MODEL_INPUT_SIZE)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=-1)
        scores[i] = float(model.predict(arr[None, ...], verbose=0)[0][0])

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise ValueError("El volumen parece uniforme o inválido: no hay slices con contraste para evaluar.")
    times["inference_seconds"] = round(time.time() - t0, 3)

    # 2) candidatas (solo válidas)
    t0 = time.time()
    cand = valid_idx[scores[valid_idx] >= THRESH_SCORE]
    if len(cand) > MAX_SLICES_GRADCAM:
        cand = cand[np.argsort(scores[cand])][::-1][:MAX_SLICES_GRADCAM]
        cand = np.sort(cand)

    # 2b) Clasificación adicional con Cheng (multicategoría), solo si el
    # llamador pasó un modelo Cheng ya cargado (solo ocurre cuando el
    # usuario tiene "Cheng" seleccionado en la app). Reutiliza los mismos
    # cortes candidatos de arriba, no añade tiempo de análisis significativo.
    cheng_result = _cheng_classify_candidate_slices(vol, cand, cheng_model, warnings)

    # 3) heat volume (pseudo heat rápido con grad_model + mean channels)
    heat = np.zeros((H, W, D), dtype=np.float32)
    for i in cand:
        s = _normalize_slice_for_model(vol[:, :, i])
        if s is None:
            continue
        img = Image.fromarray((s * 255).astype(np.uint8)).resize(MODEL_INPUT_SIZE)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.stack([arr, arr, arr], axis=-1)
        conv, _ = grad_model(arr[None, ...])
        conv_np = conv[0].numpy()
        heatmap = conv_np.mean(axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        heat_full = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H))
        heat[:, :, i] = np.array(heat_full).astype(np.float32) / 255.0

    mxh = float(heat.max())
    if mxh > 0:
        heat /= mxh
    else:
        warnings.append("Ningún slice superó el umbral de foco; el mapa de atención está vacío.")
    times["gradcam_seconds"] = round(time.time() - t0, 3)

    # 4) Normalización T1 para render 3D (percentiles 1 y 99.5)
    p1, p995 = np.percentile(vol, (1, 99.5))
    t1n = np.clip(vol, p1, p995)
    t1n = (t1n - p1) / (p995 - p1 + 1e-8)

    # 5) Downsample
    t1n_ds = t1n[::STEP_3D, ::STEP_3D, ::STEP_3D]
    heat_ds = heat[::STEP_3D, ::STEP_3D, ::STEP_3D]
    spacing_ds = (spacing_full[0] * STEP_3D, spacing_full[1] * STEP_3D, spacing_full[2] * STEP_3D)

    # 5b) Tope de resolución para el visor: si el lado sigue siendo grande,
    #     submuestreamos más (el navegador renderiza mucho más fluido).
    max_dim = max(t1n_ds.shape)
    if max_dim > VIEWER_MAX_DIM:
        extra = int(np.ceil(max_dim / float(VIEWER_MAX_DIM)))
        t1n_ds = t1n_ds[::extra, ::extra, ::extra]
        heat_ds = heat_ds[::extra, ::extra, ::extra]
        spacing_ds = tuple(s * extra for s in spacing_ds)

    # 6) Métricas / estadística (solo slices válidos)
    valid_scores = scores[valid_idx]
    order = valid_idx[np.argsort(valid_scores)[::-1]]
    top10 = [int(x) for x in order[:10]]
    best_slice = int(order[0])
    best_score = float(scores[best_slice])
    score_mean = float(np.mean(valid_scores))
    above = int(np.sum(valid_scores >= CLASSIFY_THRESHOLD))

    stats = {
        "maximum": round(float(np.max(valid_scores)), 6),
        "minimum": round(float(np.min(valid_scores)), 6),
        "mean": round(score_mean, 6),
        "median": round(float(np.median(valid_scores)), 6),
        "standard_deviation": round(float(np.std(valid_scores)), 6),
        "percentile_90": round(float(np.percentile(valid_scores, 90)), 6),
        "percentile_95": round(float(np.percentile(valid_scores, 95)), 6),
        "slices_above_threshold": above,
        "percentage_above_threshold": round(100.0 * above / len(valid_scores), 3),
    }
    # slice_scores para toda la profundidad (null donde no es válido)
    slice_scores = [None if np.isnan(v) else round(float(v), 6) for v in scores]
    top_slices = [{"index": int(idx), "score": round(float(scores[idx]), 6)} for idx in order[:5]]
    relative_pos = round(best_slice / max(1, (D - 1)), 4)

    label = "tumor" if best_score >= CLASSIFY_THRESHOLD else "no_tumor"

    # 7) Visor 3D (vtk.js) — HTML autónomo
    t0 = time.time()
    viewer_meta = {
        "filename": filename,
        "volume_shape": [H, W, D],
        "best_slice": best_slice,
        "best_score": best_score,
        "score_mean": score_mean,
        "threshold": THRESH_SCORE,
        "model_version": MODEL_VERSION,
        "request_id": request_id,
        "voxel_spacing": [round(spacing_full[0], 4), round(spacing_full[1], 4), round(spacing_full[2], 4)],
        "top_slices": top_slices,
        "statistics": stats,
        "cheng_label": (cheng_result or {}).get("cheng_label"),
        "cheng_probability": (cheng_result or {}).get("cheng_probability"),
    }
    html = build_brain_viewer_html(t1n_ds, heat_ds, spacing_ds, viewer_meta)
    html_path = f"/tmp/render3d_{int(time.time())}_{uuid.uuid4().hex[:6]}.html"
    Path(html_path).write_text(html, encoding="utf-8")
    times["viewer_generation_seconds"] = round(time.time() - t0, 3)

    times["total_seconds"] = round(time.time() - t_all, 3)

    # limpieza del NIfTI temporal
    try:
        os.remove(local_path)
    except OSError:
        pass

    return {
        "html_path": html_path,
        "filename": filename,
        "shape": (H, W, D),
        "best_slice": best_slice,
        "best_score": best_score,
        "score_mean": score_mean,
        "top10": top10,
        "top_slices": top_slices,
        "slice_scores": slice_scores,
        "slice_count": int(D),
        "valid_slice_count": int(len(valid_idx)),
        "statistics": stats,
        "voxel_spacing": [round(v, 4) for v in spacing_full],
        "relative_position": relative_pos,
        "label": label,
        "tumor_probability": round(best_score, 6),
        "no_tumor_probability": round(1.0 - best_score, 6),
        "processing_times": times,
        "warnings": warnings,
        "cheng_label": (cheng_result or {}).get("cheng_label"),
        "cheng_probability": (cheng_result or {}).get("cheng_probability"),
        "cheng_classes_probability": (cheng_result or {}).get("cheng_classes_probability"),
        "cheng_slices_used": (cheng_result or {}).get("cheng_slices_used"),
    }


# ==========================
# ENDPOINTS
# ==========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        # ---- claves originales (compatibilidad) ----
        "model_loaded": _model is not None or bool(_models_cache),
        "model_version": MODEL_VERSION,
        # ---- info detallada de ambos modelos ----
        "models_loaded": sorted(_models_cache.keys()),
        "models": {
            "binary": {
                "loaded": "binary" in _models_cache,
                "blob": BINARY_MODEL_BLOB,
                "version": BINARY_MODEL_VERSION,
            },
            "cheng": {
                "loaded": "cheng" in _models_cache,
                "blob": CHENG_MODEL_BLOB,
                "version": CHENG_MODEL_VERSION,
                "classes": CHENG_CLASSES,
            },
        },
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dependencies": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    request_id = uuid.uuid4().hex
    data = request.get_json(silent=True) or {}

    file_path = data.get("file_path") or data.get("image_path")
    if not file_path:
        return jsonify({"error": "Falta file_path (o image_path)", "request_id": request_id}), 400

    # ---- Selección de modelo (compatibilidad: por defecto "binary") ----
    model_type = str(data.get("model") or DEFAULT_MODEL_TYPE).strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        return jsonify({
            "error": f"Modelo no soportado: '{model_type}'. Usa 'binary' o 'cheng'.",
            "request_id": request_id,
        }), 400

    fp_lower = str(file_path).lower()
    is_nifti = fp_lower.endswith(ALLOWED_NIFTI_EXT)
    ext = os.path.splitext(fp_lower)[1]
    if not is_nifti and ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": f"Extensión no soportada: {ext}", "request_id": request_id}), 400

    # El pipeline NIfTI es siempre volumétrico + binario, independientemente
    # del modelo seleccionado en la app (no se ha modificado esa lógica): el
    # escaneo capa por capa y el Grad-CAM 3D siempre usan el modelo binario.
    # Si el usuario tiene "cheng" seleccionado, se usa ADEMÁS (no en vez de)
    # para clasificar el tipo de tumor sobre los cortes ya detectados como
    # sospechosos — ver _cheng_classify_candidate_slices().
    effective_model_type = "binary" if is_nifti else model_type

    try:
        model, grad_model, _conv_layer = load_model_cached(effective_model_type)
    except Exception:
        logger.exception("Error cargando modelo '%s'", effective_model_type)
        return jsonify({"error": f"No se pudo cargar el modelo '{effective_model_type}'.",
                        "request_id": request_id}), 503

    cheng_model_obj = None
    if is_nifti and model_type == "cheng":
        try:
            cheng_model_obj, _cheng_grad_model, _cheng_conv = load_model_cached("cheng")
        except Exception:
            logger.exception(
                "No se pudo cargar el modelo 'cheng' para la clasificación adicional "
                "en NIfTI; se omite esa clasificación pero el análisis binario continúa."
            )
            cheng_model_obj = None

    # ---------- NIFTI ----------
    if is_nifti:
        try:
            r = process_nifti(file_path, model, grad_model, request_id, cheng_model=cheng_model_obj)
            dest = f"renders/{os.path.basename(r['html_path'])}"
            url = upload_to_gcs(BUCKET_RENDER_3D, r["html_path"], dest, content_type="text/html; charset=utf-8")
            try:
                os.remove(r["html_path"])
            except OSError:
                pass

            _log(request_id, "nifti_ok", filename=r["filename"], times=r["processing_times"])
            resp = {
                # ---- claves originales (compatibilidad total) ----
                "type": "nifti",
                "nifti_filename": r["filename"],
                "volume_shape": [int(r["shape"][0]), int(r["shape"][1]), int(r["shape"][2])],
                "render3d_url": url,
                "best_slice": r["best_slice"],
                "best_score": r["best_score"],
                "score_mean": r["score_mean"],
                "top10_slices": r["top10"],
                # ---- claves nuevas (opcionales) ----
                "model": "binary",
                "prediction": r["label"],
                "label": "Tumor" if r["label"] == "tumor" else "NoTumor",
                "threshold": THRESH_SCORE,
                "classify_threshold": CLASSIFY_THRESHOLD,
                "tumor_probability": r["tumor_probability"],
                "no_tumor_probability": r["no_tumor_probability"],
                "slice_count": r["slice_count"],
                "valid_slice_count": r["valid_slice_count"],
                "slice_scores": r["slice_scores"],
                "top_slices": r["top_slices"],
                "score_statistics": r["statistics"],
                "relative_position": r["relative_position"],
                "voxel_spacing": r["voxel_spacing"],
                "model_version": BINARY_MODEL_VERSION,
                "request_id": request_id,
                "processing_times": r["processing_times"],
                "viewer_3d_url": url,
                "warnings": r["warnings"],
            }
            if r.get("cheng_label"):
                resp["cheng_label"] = r["cheng_label"]
                resp["cheng_probability"] = r["cheng_probability"]
                resp["cheng_classes_probability"] = r["cheng_classes_probability"]
                resp["cheng_slices_used"] = r["cheng_slices_used"]
                resp["cheng_model_version"] = CHENG_MODEL_VERSION
            return jsonify(resp), 200

        except (ValueError, FileNotFoundError) as e:
            _log(request_id, "nifti_bad_request", error=str(e))
            return jsonify({"error": str(e), "request_id": request_id}), 400
        except Exception:
            logger.exception("NIfTI failed")
            return jsonify({"error": "Fallo procesando el NIfTI.", "request_id": request_id}), 500

    # ---------- IMAGE ----------
    try:
        t_all = time.time()
        times: Dict[str, float] = {}

        t0 = time.time()
        _, blob_name = parse_gs_uri(file_path)
        ext = os.path.splitext(_safe_basename(blob_name))[1].lower() or ".jpg"
        local_img = f"/tmp/img_{uuid.uuid4().hex}{ext}"
        download_from_gcs(file_path, local_img)
        times["download_seconds"] = round(time.time() - t0, 3)

        if effective_model_type == "cheng":
            resp_json, gradcam_local_path = _predict_image_cheng(local_img, model, grad_model, times)
        else:
            resp_json, gradcam_local_path = _predict_image_binary(local_img, model, grad_model, times)

        times["total_seconds"] = round(time.time() - t_all, 3)
        resp_json["processing_times"] = times
        resp_json["request_id"] = request_id

        try:
            os.remove(local_img)
        except OSError:
            pass
        if gradcam_local_path:
            try:
                os.remove(gradcam_local_path)
            except OSError:
                pass

        _log(request_id, "image_ok", model=effective_model_type, label=resp_json.get("label"))
        return jsonify(resp_json), 200

    except (ValueError, FileNotFoundError) as e:
        _log(request_id, "image_bad_request", error=str(e))
        return jsonify({"error": str(e), "request_id": request_id}), 400
    except Exception:
        logger.exception("Image failed")
        return jsonify({"error": "Fallo procesando la imagen.", "request_id": request_id}), 500


def _predict_image_binary(local_img: str, model, grad_model, times: Dict[str, float]):
    """Pipeline de imagen para el modelo 1 (Binary Tumor Detection). Devuelve
    (json_respuesta, ruta_local_gradcam_o_None)."""
    t0 = time.time()
    img = kimage.load_img(local_img, target_size=BINARY_INPUT_SIZE)
    arr = kimage.img_to_array(img)[None, ...] / 255.0
    pred = float(model.predict(arr, verbose=0)[0][0])
    times["inference_seconds"] = round(time.time() - t0, 3)
    label = "Tumor" if pred > CLASSIFY_THRESHOLD else "NoTumor"

    gradcam_url = None
    gradcam_local_path = None
    t0 = time.time()
    if label == "Tumor" and pred >= 0.9:
        gradcam_local_path = generate_gradcam_2d(img, grad_model)
        gradcam_url = upload_to_gcs(
            BUCKET_GRADCAM_2D, gradcam_local_path,
            f"gradcam/{os.path.basename(gradcam_local_path)}",
            content_type="image/jpeg",
        )
    times["gradcam_seconds"] = round(time.time() - t0, 3)

    resp = {
        # ---- claves originales ----
        "type": "image",
        "label": label,
        "prediction": pred,
        "gradcam_url": gradcam_url,
        # ---- claves nuevas ----
        "model": "binary",
        "threshold": CLASSIFY_THRESHOLD,
        "tumor_probability": round(pred, 6),
        "no_tumor_probability": round(1.0 - pred, 6),
        "model_version": BINARY_MODEL_VERSION,
        "warnings": [],
    }
    return resp, gradcam_local_path


def _predict_image_cheng(local_img: str, model, grad_model, times: Dict[str, float]):
    """Pipeline de imagen para el modelo 2 (Cheng Brain Tumor Classification,
    multicategoría). Devuelve (json_respuesta, ruta_local_gradcam_o_None)."""
    img = kimage.load_img(local_img, target_size=CHENG_INPUT_SIZE)

    t0 = time.time()
    img_input = _preprocess_for_model(img, CHENG_INPUT_SIZE, CHENG_PREPROCESS)
    preds = model.predict(img_input, verbose=0)[0]
    times["inference_seconds"] = round(time.time() - t0, 3)

    num_classes = int(preds.shape[0])
    if len(CHENG_CLASSES) == num_classes:
        class_names = CHENG_CLASSES
    else:
        logger.warning(
            "CHENG_CLASSES tiene %d clases pero el modelo produce %d salidas. "
            "Se usan nombres genéricos (class_0, class_1, ...). Ajusta la variable "
            "de entorno CHENG_CLASSES.",
            len(CHENG_CLASSES), num_classes,
        )
        class_names = [f"class_{i}" for i in range(num_classes)]

    best_idx = int(np.argmax(preds))
    label = class_names[best_idx]
    best_prob = float(preds[best_idx])
    classes_probability = {class_names[i]: round(float(preds[i]), 6) for i in range(num_classes)}

    gradcam_url = None
    gradcam_local_path = None
    t0 = time.time()
    try:
        gradcam_local_path = generate_gradcam(
            img, grad_model, CHENG_INPUT_SIZE, class_index=best_idx, preprocess=CHENG_PREPROCESS
        )
        gradcam_url = upload_to_gcs(
            BUCKET_GRADCAM_2D, gradcam_local_path,
            f"gradcam/{os.path.basename(gradcam_local_path)}",
            content_type="image/jpeg",
        )
    except Exception:
        logger.exception("No se pudo generar Grad-CAM para el modelo Cheng")
    times["gradcam_seconds"] = round(time.time() - t0, 3)

    resp = {
        "type": "image",
        "model": "cheng",
        "label": label,
        "prediction": round(best_prob, 6),
        "classes_probability": classes_probability,
        "gradcam_url": gradcam_url,
        "model_version": CHENG_MODEL_VERSION,
        "warnings": [],
    }
    return resp, gradcam_local_path


# ==========================
# CLOUD RUN ENTRYPOINT
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


# ============================================================
# ASSETS INCRUSTADOS DEL VISOR 3D (vtk.js)
# Editar aquí el HTML/CSS/JS del visor. main.py es autónomo:
# no requiere carpetas templates/ ni static/ para desplegar.
# ============================================================
_VIEWER_CSS = """/* ============================================================
   ScanIA · Visor volumétrico 3D — tema clínico "holográfico"
   ============================================================ */
:root{
  --cyan:#5fe0ff; --cyan-soft:#1a9fd4; --amber:#ff6a4d; --red:#ff285a;
  --green:#30d158; --text:#e8f6ff; --muted:#89a9c4;
  --panel:rgba(9,20,38,0.62); --panel-solid:#0b1730;
  --border:rgba(95,224,255,0.18); --border-soft:rgba(95,224,255,0.10);
}
*{box-sizing:border-box;}
html,body{
  margin:0; height:100%; width:100%; overflow:hidden; color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  background:
    radial-gradient(1200px 820px at 50% 42%, #0a1a34 0%, #030814 55%, #01040c 100%);
}

/* Lienzo del render (vtk.js inyecta su canvas dentro) */
#viewer{position:fixed; inset:0;}
#viewer canvas{outline:none;}

/* Capas ambiente — minimizadas: solo un viñeteado sutil, sin scanlines/rejilla */
.fx{position:fixed; inset:0; pointer-events:none; z-index:1;}
.fx.vignette{box-shadow:inset 0 0 200px 60px rgba(0,0,0,0.55);}

/* Barra superior */
.topbar{position:fixed; top:0; left:0; right:0; height:56px; z-index:20;
  display:flex; align-items:center; justify-content:space-between; padding:0 20px;
  background:linear-gradient(180deg, rgba(2,8,18,0.92), rgba(2,8,18,0));
  transition:opacity .25s ease;}
.brand{display:flex; align-items:center; gap:10px; font-weight:600; letter-spacing:.3px; font-size:15px;}
.brand .dot{width:9px; height:9px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan);}
.brand small{color:var(--muted); font-weight:400; margin-left:6px; font-size:12px;}
.top-right{display:flex; align-items:center; gap:10px;}
.badge{font-size:12px; color:var(--muted); border:1px solid var(--border); padding:6px 12px;
  border-radius:20px; background:var(--panel); max-width:40vw; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap;}
.badge b{color:var(--text); font-weight:600;}

/* Paneles laterales (glass) — más planos y compactos, menos "peso" visual */
.panel{position:fixed; z-index:20; background:var(--panel);
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border:1px solid var(--border-soft); border-radius:14px; padding:12px 14px;
  box-shadow:0 12px 28px rgba(0,0,0,0.35); transition:opacity .25s ease, transform .25s ease;}
.panel h3{margin:0 0 10px; font-size:10px; letter-spacing:1.3px; color:var(--cyan);
  text-transform:uppercase; font-weight:600; display:flex; align-items:center; gap:8px;}
.panel.left{top:70px; left:16px; width:224px;}
.panel.info{bottom:16px; left:16px; width:224px;}
.panel.right{top:70px; right:16px; width:236px; max-height:calc(100vh - 150px); overflow-y:auto;}

.row{display:flex; justify-content:space-between; align-items:baseline;
  padding:6px 0; border-bottom:1px solid var(--border-soft); font-size:12px;}
.row:last-child{border-bottom:none;}
.row .k{color:var(--muted); text-transform:uppercase; letter-spacing:.5px; font-size:10px;}
.row .v{font-weight:600;} .v.cyan{color:var(--cyan);} .v.amber{color:var(--amber);} .v.green{color:var(--green);}
.row .v.violet{color:#b89bff;}

/* Chip destacado para el resultado del modelo Cheng ("tipo más probable") */
.cheng-chip{display:none; align-items:center; justify-content:space-between; gap:8px;
  margin:2px 0 10px; padding:9px 12px; border-radius:10px;
  background:rgba(184,155,255,0.10); border:1px solid rgba(184,155,255,0.28);}
.cheng-chip.show{display:flex;}
.cheng-chip .k{font-size:10px; letter-spacing:.5px; color:var(--muted); text-transform:uppercase;}
.cheng-chip .v{font-size:13px; font-weight:700; color:#b89bff;}

/* Sección plegable ("ver más") para reducir la densidad visual por defecto */
.adv-toggle{cursor:pointer; display:flex; align-items:center; justify-content:space-between;
  padding:8px 0 6px; font-size:11px; color:var(--cyan); user-select:none;}
.adv-toggle .chev{display:inline-block; transition:transform .2s ease; font-size:10px;}
.adv-toggle.open .chev{transform:rotate(90deg);}
.adv-body{display:none;}
.adv-body.open{display:block;}

.group{margin-bottom:14px;}
.group .lbl{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.6px;
  display:flex; justify-content:space-between; margin:0 0 6px;}
.group .lbl span:last-child{color:var(--cyan); font-weight:600;}

/* Controles */
input[type=range]{-webkit-appearance:none; appearance:none; width:100%; height:4px; border-radius:4px;
  background:rgba(95,224,255,.18); outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none; width:14px; height:14px; border-radius:50%;
  background:var(--cyan); cursor:pointer; box-shadow:0 0 8px rgba(95,224,255,.7);}
input[type=range]::-moz-range-thumb{width:14px; height:14px; border-radius:50%; border:none;
  background:var(--cyan); cursor:pointer;}

select{width:100%; background:var(--panel-solid); color:var(--text); border:1px solid var(--border);
  border-radius:10px; padding:8px 10px; font-size:12px; outline:none; cursor:pointer;}

.btn-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:6px;}
.btn{cursor:pointer; font-size:12px; font-weight:600; color:var(--cyan);
  background:rgba(95,224,255,0.10); border:1px solid var(--border); padding:8px 6px; border-radius:10px;
  transition:transform .1s ease, background .15s ease; text-align:center;}
.btn:hover{background:rgba(95,224,255,0.20); transform:translateY(-1px);}
.btn.primary{color:#02121f; background:var(--cyan); box-shadow:0 0 16px rgba(95,224,255,.5); border:none;}
.btn.primary:hover{box-shadow:0 0 24px rgba(95,224,255,.75);}
.btn.wide{grid-column:1 / -1;}
.btn.active{background:rgba(95,224,255,0.28); color:#eaffff;}

.switch{display:flex; align-items:center; justify-content:space-between; padding:7px 0; font-size:12px; color:var(--muted);}
.toggle{position:relative; width:38px; height:20px; border-radius:20px; background:rgba(95,224,255,.16);
  cursor:pointer; transition:background .2s ease;}
.toggle::after{content:''; position:absolute; top:2px; left:2px; width:16px; height:16px; border-radius:50%;
  background:#7fa8c4; transition:transform .2s ease, background .2s ease;}
.toggle.on{background:rgba(95,224,255,.5);}
.toggle.on::after{transform:translateX(18px); background:var(--cyan);}

/* Leyenda */
.legend .grad{height:10px; border-radius:6px; margin:2px 0 6px;
  background:linear-gradient(90deg,#08305c,#0e6ca8,#1eaad6,#60d6ec,#e8fcff);}
.legend .glabels{display:flex; justify-content:space-between; font-size:10px; color:var(--muted);}
.legend .focus{display:flex; align-items:center; gap:9px; margin-top:10px; font-size:11px; color:var(--muted);}
.legend .sw{width:26px; height:10px; border-radius:5px;
  background:linear-gradient(90deg,#ff6a4d,#ff285a); box-shadow:0 0 10px rgba(255,60,90,.6);}

/* Barra inferior de acciones */
.dock{position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:20;
  display:flex; gap:8px; align-items:center; background:var(--panel);
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border:1px solid var(--border-soft); border-radius:22px; padding:8px 10px;
  box-shadow:0 12px 28px rgba(0,0,0,.35); transition:opacity .25s ease;}
.dock .ico{width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  cursor:pointer; color:var(--cyan); background:rgba(95,224,255,.08); border:1px solid var(--border-soft);
  font-size:16px; transition:background .15s ease, transform .1s ease;}
.dock .ico:hover{background:rgba(95,224,255,.2); transform:translateY(-2px);}
.dock .ico.active{background:rgba(95,224,255,.3);}
.dock .sep{width:1px; height:24px; background:var(--border-soft);}

/* Toast / hint */
.hint{position:fixed; bottom:74px; left:50%; transform:translateX(-50%); z-index:20;
  font-size:11px; color:var(--muted); background:var(--panel); border:1px solid var(--border-soft);
  padding:6px 12px; border-radius:16px; opacity:.85;}

/* Loader */
.loader{position:fixed; inset:0; z-index:40; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:18px; background:rgba(1,4,12,.85);}
.spinner{width:54px; height:54px; border-radius:50%; border:3px solid rgba(95,224,255,.15);
  border-top-color:var(--cyan); animation:spin 1s linear infinite;}
.loader .t{font-size:13px; color:var(--muted); letter-spacing:.5px;}
@keyframes spin{to{transform:rotate(360deg);}}

/* Error WebGL */
.errbox{position:fixed; inset:0; z-index:50; display:none; flex-direction:column;
  align-items:center; justify-content:center; gap:14px; background:rgba(1,4,12,.95); text-align:center; padding:24px;}
.errbox.show{display:flex;}
.errbox .icon{font-size:44px;}
.errbox h2{margin:0; font-size:18px;}
.errbox p{max-width:440px; color:var(--muted); font-size:13px; line-height:1.5;}

/* Disclaimer */
.disclaimer{position:fixed; bottom:6px; left:50%; transform:translateX(-50%); z-index:15;
  font-size:10px; color:var(--muted); opacity:.55; max-width:80vw; text-align:center;}

.hidden-ui .topbar,.hidden-ui .panel,.hidden-ui .dock,.hidden-ui .hint,.hidden-ui .disclaimer{opacity:0; pointer-events:none;}

@media (max-width:820px){
  .panel.left,.panel.right,.panel.info{display:none;}
  .badge{max-width:50vw;}
}
"""

_VIEWER_JS = """/* ============================================================
   ScanIA · Visor volumétrico 3D (vtk.js)
   Renderizado volumétrico GPU con transfer functions, presets,
   overlay Grad-CAM, planos de recorte, vistas médicas y captura.
   Lee el volumen inyectado en window.SCANIA_VOLUME.
   ============================================================ */
(function () {
  "use strict";

  var DATA = window.SCANIA_VOLUME || {};
  var META = DATA.meta || {};

  // -------- WebGL disponible? --------
  function hasWebGL() {
    try {
      var c = document.createElement("canvas");
      return !!(window.WebGLRenderingContext &&
        (c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl")));
    } catch (e) { return false; }
  }

  function showError(msg) {
    var box = document.getElementById("errbox");
    if (box) { box.querySelector("p").textContent = msg; box.classList.add("show"); }
    var loader = document.getElementById("loader");
    if (loader) loader.style.display = "none";
  }

  if (!window.vtk) { showError("No se pudo cargar la librería de renderizado (vtk.js). Comprueba tu conexión."); return; }
  if (!hasWebGL()) { showError("Tu navegador o equipo no tiene WebGL disponible. El visor volumétrico requiere aceleración gráfica."); return; }

  // -------- utilidades --------
  function b64ToU8(b64) {
    if (!b64) return new Uint8Array(0);
    var bin = atob(b64), len = bin.length, u8 = new Uint8Array(len);
    for (var i = 0; i < len; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }

  var dims = DATA.dims || [1, 1, 1];
  var spacing = DATA.spacing || [1, 1, 1];
  var t1arr = b64ToU8(DATA.t1);
  var heatArr = DATA.hasHeat ? b64ToU8(DATA.heat) : null;
  var heatShow = (DATA.heatShow != null ? DATA.heatShow : 0.45) * 255.0;

  // ============================================================
  // vtk.js setup
  // ============================================================
  var V = window.vtk;
  var grw = V.Rendering.Misc.vtkGenericRenderWindow.newInstance({ background: [0, 0, 0, 0] });
  grw.setContainer(document.getElementById("viewer"));
  var renderer = grw.getRenderer();
  var renderWindow = grw.getRenderWindow();
  // Redimensionado con tope de densidad de píxeles (evita 4x coste en pantallas Retina/4K)
  function doResize() {
    try {
      var el = document.getElementById("viewer");
      var dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      var w = Math.max(1, Math.floor(el.clientWidth * dpr));
      var h = Math.max(1, Math.floor(el.clientHeight * dpr));
      var arw = (grw.getApiSpecificRenderWindow && grw.getApiSpecificRenderWindow()) ||
                (grw.getOpenGLRenderWindow && grw.getOpenGLRenderWindow());
      if (arw && arw.setSize) { arw.setSize(w, h); renderWindow.render(); return; }
    } catch (e) {}
    grw.resize();
  }
  doResize();
  window.addEventListener("resize", doResize);

  try {
    var style = V.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance();
    renderWindow.getInteractor().setInteractorStyle(style);
  } catch (e) { /* estilo por defecto */ }

  var interactor = renderWindow.getInteractor();
  try {
    // Durante la interacción se baja la calidad automáticamente -> fluido.
    // En reposo se hace un render de alta calidad.
    interactor.setDesiredUpdateRate(15.0);
    interactor.setStillUpdateRate(0.6);
  } catch (e) {}

  // -------- ImageData T1 --------
  function makeImageData(values) {
    var img = V.Common.DataModel.vtkImageData.newInstance();
    img.setDimensions(dims[0], dims[1], dims[2]);
    img.setSpacing(spacing[0], spacing[1], spacing[2]);
    img.setOrigin(0, 0, 0);
    var da = V.Common.Core.vtkDataArray.newInstance({
      name: "scalars", numberOfComponents: 1, values: values,
    });
    img.getPointData().setScalars(da);
    return img;
  }

  var t1Image = makeImageData(t1arr);

  var mapper = V.Rendering.Core.vtkVolumeMapper.newInstance();
  mapper.setInputData(t1Image);
  mapper.setSampleDistance(1.1);            // equilibrio nitidez/rendimiento
  mapper.setMaximumSamplesPerRay(1200);
  if (mapper.setAutoAdjustSampleDistances) mapper.setAutoAdjustSampleDistances(true);

  var actor = V.Rendering.Core.vtkVolume.newInstance();
  actor.setMapper(mapper);

  var ctf = V.Rendering.Core.vtkColorTransferFunction.newInstance();
  var otf = V.Common.DataModel.vtkPiecewiseFunction.newInstance();
  var prop = actor.getProperty();
  prop.setRGBTransferFunction(0, ctf);
  prop.setScalarOpacity(0, otf);
  prop.setInterpolationTypeToLinear();
  prop.setUseGradientOpacity(0, false);      // off por rendimiento
  prop.setScalarOpacityUnitDistance(0, 1.4); // menor => más "cuerpo", menos niebla
  prop.setShade(true);
  prop.setAmbient(0.4);
  prop.setDiffuse(0.85);
  prop.setSpecular(0.35);
  prop.setSpecularPower(12.0);
  renderer.addVolume(actor);

  // -------- Overlay Grad-CAM --------
  var heatActor = null, heatCtf = null, heatOtf = null, heatMapper = null, heatImage = null;
  if (heatArr) {
    heatImage = makeImageData(heatArr);
    heatMapper = V.Rendering.Core.vtkVolumeMapper.newInstance();
    heatMapper.setInputData(heatImage);
    heatMapper.setSampleDistance(1.6);
    if (heatMapper.setAutoAdjustSampleDistances) heatMapper.setAutoAdjustSampleDistances(true);
    heatActor = V.Rendering.Core.vtkVolume.newInstance();
    heatActor.setMapper(heatMapper);
    heatCtf = V.Rendering.Core.vtkColorTransferFunction.newInstance();
    heatOtf = V.Common.DataModel.vtkPiecewiseFunction.newInstance();
    var hp = heatActor.getProperty();
    hp.setRGBTransferFunction(0, heatCtf);
    hp.setScalarOpacity(0, heatOtf);
    hp.setInterpolationTypeToLinear();
    hp.setShade(false);
    // No se añade al renderer aún: se retrasa hasta justo después del primer
    // frame (ver "Arranque") para que el volumen T1 se pinte cuanto antes y
    // no se paguen dos pipelines de shader/textura a la vez en la carga.
  }

  // ============================================================
  // Presets (color/opacidad normalizados 0..1) + estado W/L/opacidad
  // ============================================================
  var PRESETS = {
    grayscale: { color: [[0, 12, 14, 18], [0.35, 90, 98, 112], [0.7, 190, 198, 212], [1, 250, 252, 255]],
                 opacity: [[0.16, 0], [0.32, 0.10], [0.6, 0.45], [1, 0.85]] },
    soft:      { color: [[0, 16, 30, 46], [0.33, 55, 130, 155], [0.68, 140, 205, 214], [1, 240, 250, 255]],
                 opacity: [[0.15, 0], [0.32, 0.10], [0.6, 0.42], [1, 0.82]] },
    highcontrast:{ color: [[0, 2, 10, 22], [0.42, 20, 140, 190], [0.7, 120, 220, 245], [1, 255, 255, 255]],
                 opacity: [[0.20, 0], [0.36, 0.06], [0.55, 0.5], [1, 0.92]] },
    cyan:      { color: [[0, 6, 22, 42], [0.32, 22, 118, 176], [0.56, 46, 182, 220], [0.8, 128, 224, 244], [1, 244, 253, 255]],
                 opacity: [[0.15, 0], [0.32, 0.12], [0.6, 0.44], [1, 0.86]] },
    warm:      { color: [[0, 24, 12, 6], [0.38, 155, 78, 34], [0.7, 228, 158, 66], [1, 255, 246, 218]],
                 opacity: [[0.15, 0], [0.34, 0.12], [0.68, 0.45], [1, 0.84]] },
  };

  var state = {
    preset: "cyan",
    level: 0.5,      // 0..1 centro de ventana
    window: 0.9,     // 0..1 ancho de ventana
    opacity: 0.9,    // multiplicador de opacidad
    autoRotate: false,
    rotateSpeed: 0.6,
    showAxes: true,
    showGrad: !!heatArr,
    gradOpacity: 0.55,
    gradThresh: 0.55,
  };

  function applyTransfer() {
    var p = PRESETS[state.preset] || PRESETS.cyan;
    ctf.removeAllPoints();
    otf.removeAllPoints();
    // Ventana/nivel: mapea pos(0..1) -> escalar(0..255) recortando por [lo,hi]
    var lo = Math.max(0, (state.level - state.window / 2));
    var hi = Math.min(1, (state.level + state.window / 2));
    function toScalar(pos) { return (lo + pos * (hi - lo)) * 255.0; }
    p.color.forEach(function (c) { ctf.addRGBPoint(toScalar(c[0]), c[1] / 255, c[2] / 255, c[3] / 255); });
    p.opacity.forEach(function (o) { otf.addPoint(toScalar(o[0]), o[1] * state.opacity); });
    renderWindow.render();
  }

  function applyHeatTransfer() {
    if (!heatArr) return;
    heatCtf.removeAllPoints();
    heatOtf.removeAllPoints();
    // colorscale ámbar -> rojo (brillante)
    heatCtf.addRGBPoint(0, 1.0, 0.58, 0.22);
    heatCtf.addRGBPoint(160, 1.0, 0.36, 0.24);
    heatCtf.addRGBPoint(255, 1.0, 0.22, 0.42);
    // corte duro por umbral: solo lo más "caliente" es visible -> foco, no losa
    var thr = Math.max(1, Math.min(240, state.gradThresh * 255.0));
    var mid = Math.min(254, (thr + 255) / 2);
    heatOtf.addPoint(0, 0.0);
    heatOtf.addPoint(thr, 0.0);
    heatOtf.addPoint(mid, 0.4 * state.gradOpacity);
    heatOtf.addPoint(255, 0.9 * state.gradOpacity);
    heatActor.setVisibility(state.showGrad);
    renderWindow.render();
  }

  // ============================================================
  // Ejes / orientación
  // ============================================================
  var omw = null;
  try {
    var axes = V.Rendering.Core.vtkAxesActor.newInstance();
    omw = V.Interaction.Widgets.vtkOrientationMarkerWidget.newInstance({
      actor: axes, interactor: renderWindow.getInteractor(),
    });
    omw.setEnabled(true);
    omw.setViewportCorner(V.Interaction.Widgets.vtkOrientationMarkerWidget.Corners.BOTTOM_RIGHT);
    omw.setViewportSize(0.12);
    omw.setMinPixelSize(60);
    omw.setMaxPixelSize(160);
  } catch (e) { omw = null; }

  // ============================================================
  // Planos de recorte (axial / coronal / sagital)
  // ============================================================
  var clip = { 0: { on: false, frac: 0.5 }, 1: { on: false, frac: 0.5 }, 2: { on: false, frac: 0.5 } };
  var normals = { 0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1] };

  function rebuildClips() {
    mapper.removeAllClippingPlanes();
    if (heatMapper) heatMapper.removeAllClippingPlanes();
    var b = t1Image.getBounds(); // [xmin,xmax,ymin,ymax,zmin,zmax]
    Object.keys(clip).forEach(function (ax) {
      var a = parseInt(ax, 10);
      if (!clip[a].on) return;
      var lo = b[a * 2], hi = b[a * 2 + 1];
      var pos = lo + clip[a].frac * (hi - lo);
      var origin = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
      origin[a] = pos;
      var plane = V.Common.DataModel.vtkPlane.newInstance({ origin: origin, normal: normals[a] });
      mapper.addClippingPlane(plane);
      if (heatMapper) {
        var plane2 = V.Common.DataModel.vtkPlane.newInstance({ origin: origin.slice(), normal: normals[a] });
        heatMapper.addClippingPlane(plane2);
      }
    });
    renderWindow.render();
  }

  // ============================================================
  // Cámara: reset, vistas, auto-rotación
  // ============================================================
  function bounds() { return t1Image.getBounds(); }
  function center() { var b = bounds(); return [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]; }
  function maxDim() { var b = bounds(); return Math.max(b[1] - b[0], b[3] - b[2], b[5] - b[4]); }

  function setView(name) {
    var c = center(), dist = 2.1 * maxDim(), cam = renderer.getActiveCamera();
    var pos = {
      anterior: [c[0], c[1] - dist, c[2]], posterior: [c[0], c[1] + dist, c[2]],
      left: [c[0] - dist, c[1], c[2]], right: [c[0] + dist, c[1], c[2]],
      superior: [c[0], c[1], c[2] + dist], inferior: [c[0], c[1], c[2] - dist],
    }[name];
    cam.setFocalPoint(c[0], c[1], c[2]);
    cam.setPosition(pos[0], pos[1], pos[2]);
    cam.setViewUp((name === "superior" || name === "inferior") ? [0, 1, 0] : [0, 0, 1]);
    renderer.resetCameraClippingRange();
    renderWindow.render();
  }

  function resetCamera() {
    renderer.resetCamera();
    var cam = renderer.getActiveCamera();
    cam.elevation(-15); cam.azimuth(30);
    renderer.resetCameraClippingRange();
    renderWindow.render();
  }

  var ROT_TOKEN = { rot: true };
  try {
    interactor.onAnimation(function () {
      if (state.autoRotate) {
        renderer.getActiveCamera().azimuth(state.rotateSpeed);
        renderer.resetCameraClippingRange();
      }
    });
  } catch (e) {}

  function setAutoRotate(on) {
    state.autoRotate = on;
    try {
      if (on) interactor.requestAnimation(ROT_TOKEN);
      else interactor.cancelAnimation(ROT_TOKEN);
    } catch (e) {
      // Fallback si la API de animación no existe: bucle manual throttled
      if (on) { (function loop(){ if(!state.autoRotate) return;
        renderer.getActiveCamera().azimuth(state.rotateSpeed); renderer.resetCameraClippingRange();
        renderWindow.render(); setTimeout(function(){ requestAnimationFrame(loop); }, 45); })(); }
    }
    var b = document.getElementById("d_rotate"); if (b) b.classList.toggle("active", on);
  }

  // ============================================================
  // Captura / pantalla completa / UI
  // ============================================================
  function screenshot() {
    try {
      renderWindow.captureImages()[0].then(function (png) {
        var a = document.createElement("a");
        a.href = png; a.download = "scania_3d_" + Date.now() + ".png"; a.click();
      });
    } catch (e) { console.error(e); }
  }

  function toggleFullscreen() {
    var el = document.documentElement;
    if (!document.fullscreenElement) { (el.requestFullscreen || el.webkitRequestFullscreen).call(el); }
    else { (document.exitFullscreen || document.webkitExitFullscreen).call(document); }
  }

  var uiHidden = false;
  function toggleUI() { uiHidden = !uiHidden; document.body.classList.toggle("hidden-ui", uiHidden); }

  // ============================================================
  // Cableado de la interfaz
  // ============================================================
  function $(id) { return document.getElementById(id); }
  function on(id, ev, fn) { var e = $(id); if (e) e.addEventListener(ev, fn); }
  function bindRange(id, key, transform) {
    var el = $(id); if (!el) return;
    el.addEventListener("input", function () {
      var val = parseFloat(el.value);
      state[key] = transform ? transform(val) : val;
      var out = $(id + "_val"); if (out) out.textContent = el.value;
      if (key === "gradOpacity" || key === "gradThresh") applyHeatTransfer(); else applyTransfer();
    });
  }

  function fillMeta() {
    function set(id, v) { var e = $(id); if (e) e.textContent = v; }
    set("badge_file", META.filename || "Estudio 3D");
    set("badge_model", META.model_version || "");
    set("m_file", META.filename || "—");
    set("m_shape", (META.volume_shape || []).join(" · "));
    set("m_best", META.best_slice != null ? META.best_slice : "—");
    set("m_score", META.best_score != null ? (META.best_score * 100).toFixed(2) + "%" : "—");
    set("m_mean", META.score_mean != null ? (META.score_mean * 100).toFixed(2) + "%" : "—");
    set("m_spacing", (META.voxel_spacing || []).map(function (x) { return x.toFixed(2); }).join(" · ") + " mm");
    set("m_model", META.model_version || "—");
    var st = META.statistics || {};
    set("s_max", st.maximum != null ? (st.maximum * 100).toFixed(1) + "%" : "—");
    set("s_median", st.median != null ? (st.median * 100).toFixed(1) + "%" : "—");
    set("s_p95", st.percentile_95 != null ? (st.percentile_95 * 100).toFixed(1) + "%" : "—");
    set("s_above", st.slices_above_threshold != null ? st.slices_above_threshold : "—");

    // Chip "tipo más probable" (modelo Cheng) — solo si el backend lo envió
    // (solo ocurre cuando el usuario tenía "Cheng" seleccionado en la app).
    var chengChip = $("cheng_chip");
    if (chengChip) {
      if (META.cheng_label) {
        var pct = META.cheng_probability != null ? " (" + (META.cheng_probability * 100).toFixed(1) + "%)" : "";
        set("m_cheng", META.cheng_label + pct);
        chengChip.classList.add("show");
      } else {
        chengChip.classList.remove("show");
      }
    }
  }

  function wireCollapsibles() {
    [["adv_toggle_left", "adv_body_left"], ["adv_toggle_right", "adv_body_right"],
     ["adv_toggle_info", "adv_body_info"]].forEach(function (pair) {
      var toggle = $(pair[0]), body = $(pair[1]);
      if (!toggle || !body) return;
      toggle.addEventListener("click", function () {
        var open = !body.classList.contains("open");
        body.classList.toggle("open", open);
        toggle.classList.toggle("open", open);
      });
    });
  }

  function wireUI() {
    // Preset
    on("preset", "change", function () { state.preset = $("preset").value; applyTransfer(); });
    // Rangos T1
    bindRange("opacity", "opacity", function (v) { return v / 100; });
    bindRange("level", "level", function (v) { return v / 100; });
    bindRange("window", "window", function (v) { return v / 100; });
    // Grad-CAM
    bindRange("gradOpacity", "gradOpacity", function (v) { return v / 100; });
    bindRange("gradThresh", "gradThresh", function (v) { return v / 100; });
    // Auto-rotación
    bindRange("speed", "rotateSpeed", function (v) { return v / 100; });
    // Vistas
    ["anterior", "posterior", "left", "right", "superior", "inferior"].forEach(function (name) {
      on("view_" + name, "click", function () { setView(name); });
    });
    on("view_reset", "click", resetCamera);
    // Dock
    on("d_rotate", "click", function () { setAutoRotate(!state.autoRotate); });
    on("d_screenshot", "click", screenshot);
    on("d_fullscreen", "click", toggleFullscreen);
    on("d_ui", "click", toggleUI);
    on("d_axes", "click", function () {
      state.showAxes = !state.showAxes;
      if (omw) omw.setEnabled(state.showAxes);
      $("d_axes").classList.toggle("active", state.showAxes);
      renderWindow.render();
    });
    // Toggles de recorte
    [0, 1, 2].forEach(function (ax) {
      on("clip_toggle_" + ax, "click", function () {
        clip[ax].on = !clip[ax].on;
        $("clip_toggle_" + ax).classList.toggle("on", clip[ax].on);
        rebuildClips();
      });
      var sl = $("clip_" + ax);
      if (sl) sl.addEventListener("input", function () { clip[ax].frac = parseFloat(sl.value) / 100; if (clip[ax].on) rebuildClips(); });
    });
    // Toggle Grad-CAM
    on("grad_toggle", "click", function () {
      state.showGrad = !state.showGrad;
      $("grad_toggle").classList.toggle("on", state.showGrad);
      applyHeatTransfer();
    });

    // Atajos de teclado
    window.addEventListener("keydown", function (e) {
      switch (e.key.toLowerCase()) {
        case "r": setAutoRotate(!state.autoRotate); break;
        case "h": toggleUI(); break;
        case "f": toggleFullscreen(); break;
        case "s": screenshot(); break;
        case "0": resetCamera(); break;
      }
    });
  }

  // ============================================================
  // Arranque
  // ============================================================
  fillMeta();
  wireUI();
  wireCollapsibles();
  applyTransfer();
  if (omw) omw.setEnabled(state.showAxes);
  resetCamera();
  renderWindow.render();
  // Auto-rotación desactivada por defecto (rendimiento). Se activa con 🌀 o tecla "R".

  // El volumen T1 ya está pintado; ahora se añade el overlay Grad-CAM (si lo
  // hay) y se aplica su transferencia. Retrasarlo un frame evita compilar y
  // subir dos pipelines de volumen a la vez, así el cerebro aparece antes.
  if (heatActor) {
    requestAnimationFrame(function () {
      renderer.addVolume(heatActor);
      applyHeatTransfer();
    });
  }

  // Durante el arrastre (rotar/zoom/pan) se oculta el overlay Grad-CAM para
  // que solo se raycastee un volumen -> interacción fluida sin tirones. Al
  // soltar, se restaura con un render de calidad completa.
  try {
    interactor.onStartInteractionEvent(function () {
      if (heatActor && state.showGrad) heatActor.setVisibility(false);
    });
    interactor.onEndInteractionEvent(function () {
      if (heatActor && state.showGrad) heatActor.setVisibility(true);
      renderWindow.render();
    });
  } catch (e) {}

  // Ocultar loader cuando el primer render está listo
  setTimeout(function () { var l = $("loader"); if (l) l.style.display = "none"; }, 250);

})();
"""

_VIEWER_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
  <title>ScanIA · Visor volumétrico 3D</title>
  <style>__CSS__</style>
  <script src="https://unpkg.com/vtk.js"></script>
</head>
<body>
  <!-- Lienzo del render -->
  <div id="viewer"></div>

  <!-- Capa de ambiente (minimizada: solo viñeteado sutil) -->
  <div class="fx vignette"></div>

  <!-- Barra superior -->
  <div class="topbar">
    <div class="brand"><span class="dot"></span>ScanIA
      <small>Visor volumétrico 3D</small></div>
    <div class="top-right">
      <div class="badge">📄 <b id="badge_file">Estudio 3D</b></div>
      <div class="badge"><span id="badge_model"></span></div>
    </div>
  </div>

  <!-- Panel izquierdo: apariencia (compacto por defecto) -->
  <div class="panel left">
    <h3>🎨 Apariencia</h3>
    <div class="group">
      <div class="lbl"><span>Preset visual</span></div>
      <select id="preset">
        <option value="cyan" selected>Cyan hologram</option>
        <option value="grayscale">MRI grayscale</option>
        <option value="soft">Soft tissue</option>
        <option value="highcontrast">High contrast</option>
        <option value="warm">Warm tissue</option>
      </select>
    </div>
    <div class="switch">Mostrar atención del modelo (Grad-CAM)
      <div class="toggle on" id="grad_toggle"></div>
    </div>

    <div class="adv-toggle" id="adv_toggle_left"><span>Más ajustes de imagen</span><span class="chev">▶</span></div>
    <div class="adv-body" id="adv_body_left">
      <div class="group">
        <div class="lbl"><span>Opacidad</span><span id="opacity_val">90</span></div>
        <input type="range" id="opacity" min="0" max="100" value="90" />
      </div>
      <div class="group">
        <div class="lbl"><span>Brillo (nivel)</span><span id="level_val">50</span></div>
        <input type="range" id="level" min="0" max="100" value="50" />
      </div>
      <div class="group">
        <div class="lbl"><span>Contraste (ventana)</span><span id="window_val">90</span></div>
        <input type="range" id="window" min="10" max="100" value="90" />
      </div>
      <div class="group">
        <div class="lbl"><span>Opacidad Grad-CAM</span><span id="gradOpacity_val">55</span></div>
        <input type="range" id="gradOpacity" min="0" max="100" value="55" />
      </div>
      <div class="group">
        <div class="lbl"><span>Umbral visual Grad-CAM</span><span id="gradThresh_val">55</span></div>
        <input type="range" id="gradThresh" min="0" max="100" value="55" />
      </div>
    </div>
  </div>

  <!-- Panel derecho: cámara y cortes (compacto por defecto) -->
  <div class="panel right">
    <h3>🎥 Vistas</h3>
    <div class="btn-grid">
      <div class="btn" id="view_anterior">Anterior</div>
      <div class="btn" id="view_posterior">Posterior</div>
      <div class="btn" id="view_left">Izq.</div>
      <div class="btn" id="view_right">Der.</div>
      <div class="btn" id="view_superior">Superior</div>
      <div class="btn" id="view_inferior">Inferior</div>
      <div class="btn primary wide" id="view_reset">Restablecer cámara</div>
    </div>

    <div class="adv-toggle" id="adv_toggle_right"><span>Rotación y planos de recorte</span><span class="chev">▶</span></div>
    <div class="adv-body" id="adv_body_right">
      <h3 style="margin-top:6px;">🌀 Rotación</h3>
      <div class="group">
        <div class="lbl"><span>Velocidad</span><span id="speed_val">40</span></div>
        <input type="range" id="speed" min="0" max="200" value="40" />
      </div>

      <h3 style="margin-top:6px;">✂️ Planos de recorte</h3>
      <div class="switch">Sagital (X)
        <div class="toggle" id="clip_toggle_0"></div>
      </div>
      <input type="range" id="clip_0" min="0" max="100" value="50" />
      <div class="switch" style="margin-top:6px;">Coronal (Y)
        <div class="toggle" id="clip_toggle_1"></div>
      </div>
      <input type="range" id="clip_1" min="0" max="100" value="50" />
      <div class="switch" style="margin-top:6px;">Axial (Z)
        <div class="toggle" id="clip_toggle_2"></div>
      </div>
      <input type="range" id="clip_2" min="0" max="100" value="50" />
    </div>
  </div>

  <!-- Panel inferior-izquierdo: información esencial (el resto, plegado) -->
  <div class="panel info legend">
    <h3>📊 Información</h3>
    <div class="cheng-chip" id="cheng_chip">
      <span class="k">Tipo más probable</span><span class="v" id="m_cheng">—</span>
    </div>
    <div class="row"><span class="k">Archivo</span><span class="v" id="m_file">—</span></div>
    <div class="row"><span class="k">Volumen</span><span class="v cyan" id="m_shape">—</span></div>
    <div class="row"><span class="k">Mejor slice</span><span class="v" id="m_best">—</span></div>
    <div class="row"><span class="k">Confianza máx.</span><span class="v amber" id="m_score">—</span></div>

    <div class="adv-toggle" id="adv_toggle_info"><span>Ver más estadísticas</span><span class="chev">▶</span></div>
    <div class="adv-body" id="adv_body_info">
      <div class="row"><span class="k">Score medio</span><span class="v" id="m_mean">—</span></div>
      <div class="row"><span class="k">Máximo</span><span class="v" id="s_max">—</span></div>
      <div class="row"><span class="k">Mediana</span><span class="v" id="s_median">—</span></div>
      <div class="row"><span class="k">P95</span><span class="v" id="s_p95">—</span></div>
      <div class="row"><span class="k">Slices &gt; umbral</span><span class="v" id="s_above">—</span></div>
      <div class="row"><span class="k">Vóxel</span><span class="v" id="m_spacing">—</span></div>
      <div class="row"><span class="k">Modelo</span><span class="v" id="m_model">—</span></div>
      <div style="margin-top:12px;">
        <div class="lbl" style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;">Densidad tisular</div>
        <div class="grad"></div>
        <div class="glabels"><span>Fluido</span><span>Tejido</span><span>Hueso</span></div>
        <div class="focus"><span class="sw"></span>Atención del modelo (Grad-CAM)</div>
      </div>
    </div>
  </div>

  <!-- Dock inferior -->
  <div class="dock">
    <div class="ico" id="d_rotate" title="Rotación automática (R)">🌀</div>
    <div class="ico active" id="d_axes" title="Ejes de orientación">🧭</div>
    <div class="sep"></div>
    <div class="ico" id="d_screenshot" title="Captura (S)">📷</div>
    <div class="ico" id="d_fullscreen" title="Pantalla completa (F)">⛶</div>
    <div class="ico" id="d_ui" title="Ocultar interfaz (H)">👁</div>
  </div>

  <div class="hint">Arrastra para rotar · Rueda para zoom · R rotación · H interfaz · F pantalla completa · S captura</div>

  <div class="disclaimer">
    ScanIA es una herramienta experimental de apoyo. El resultado no constituye un diagnóstico médico y debe ser interpretado por un profesional sanitario cualificado. Grad-CAM es explicabilidad, no una segmentación tumoral exacta.
  </div>

  <!-- Loader -->
  <div class="loader" id="loader">
    <div class="spinner"></div>
    <div class="t">Reconstruyendo volumen 3D…</div>
  </div>

  <!-- Error WebGL -->
  <div class="errbox" id="errbox">
    <div class="icon">🧊</div>
    <h2>No se puede mostrar el visor 3D</h2>
    <p>El visor volumétrico requiere WebGL / aceleración gráfica.</p>
  </div>

  <script>window.SCANIA_VOLUME = __VOLUME__;</script>
  <script>__JS__</script>
</body>
</html>
"""
