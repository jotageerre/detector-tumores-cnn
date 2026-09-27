"""
entrenar_modelo_final.py
========================

REIMPLEMENTACIÓN del protocolo de evaluación final del TFG (Anexo C.1 /
docs/protocolos/C1_evaluacion_final.md), AÑADIDA para el repositorio público.

Importante: el notebook exacto que ejecutó el entrenamiento final (job 70930,
balanced accuracy 93,89 % en test) no se conserva entre los artefactos
localizados (ver Anexo A.3). Este script reconstruye ese protocolo a partir de:
  - la configuración escrita en el protocolo C.1, y
  - el código real de training/robustez_5cv/ejecutar_fold_5cv_robustez.py,
    que implementa EXACTAMENTE la misma receta (fase cabeza 6 épocas Adam 1e-4
    + fine-tuning 6 épocas Adam 1e-5 de las últimas 20 capas no-BN).
Por falta de determinismo estricto en GPU, una reejecución no reproducirá las
cifras al decimal; debería quedar en el mismo orden de magnitud.

Entrena ResNet50-FT sobre los 198 pacientes de train+val del split congelado,
guarda el modelo (.keras y .h5, este último compatible con el backend de
Cloud Run) y lo evalúa UNA SOLA VEZ sobre los 35 pacientes de test.

Uso:
    python entrenar_modelo_final.py --root ./tfg_run
Requiere <root>/results/artifacts/manifest_final.csv
(créalo con herramientas/construir_manifest_cheng.py).
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    recall_score, precision_score, confusion_matrix,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="tfg_run")
args = ap.parse_args()

ROOT = Path(args.root).resolve()
MANIFEST = ROOT / "results/artifacts/manifest_final.csv"
OUT = ROOT / "results/tumor_type/final"
OUT.mkdir(parents=True, exist_ok=True)

# --- configuración fijada en el protocolo C.1 --------------------------------
CLASSES = ["glioma", "meningioma", "pituitario"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD, EPOCHS_FINETUNE = 6, 6
LR_HEAD, LR_FINETUNE = 1e-4, 1e-5
N_CAPAS_FINETUNE = 20
SEED = 42
SPLIT_SHA256 = "fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092"
EXPECTED_TOTAL_PARAMS = 23850371
EXPECTED_HEAD_TRAINABLE_PARAMS = 262659
EXPECTED_FT_TRAINABLE_PARAMS = 14691331

STARTED = OUT / "FINAL_TEST_STARTED.lock"
COMPLETED = OUT / "FINAL_TEST_COMPLETED.lock"
if STARTED.exists() or COMPLETED.exists():
    raise RuntimeError("ABORTADO: el test ya se abrió en esta carpeta. "
                       "El protocolo permite evaluar el test UNA sola vez.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def n_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


# --- datos y verificación del split congelado --------------------------------
data = pd.read_csv(MANIFEST)
cheng = data[data["source_dataset"] == "figshare/Cheng2017"].reset_index(drop=True)
assert len(cheng) == 3064 and cheng["patient_id"].nunique() == 233

pac = (cheng.groupby("patient_id")
       .agg(subset=("subset", "first"), tumor_type=("tumor_type", "first"))
       .reset_index().sort_values("patient_id").reset_index(drop=True))
huella = hashlib.sha256(pac[["patient_id", "subset", "tumor_type"]]
                        .to_csv(index=False).encode("utf-8")).hexdigest()
assert huella == SPLIT_SHA256, f"split distinto del congelado: {huella}"

train = cheng[cheng["subset"].isin(["train", "val"])].reset_index(drop=True)
test = cheng[cheng["subset"] == "test"].reset_index(drop=True)
assert train["patient_id"].nunique() == 198 and test["patient_id"].nunique() == 35
assert set(train["patient_id"]).isdisjoint(set(test["patient_id"]))
print(f"TRAIN: 198 pacientes / {len(train)} cortes | TEST: 35 pacientes / {len(test)} cortes")

# --- pipeline tf.data (idéntico a ejecutar_fold_5cv_robustez.py) -------------
keras.utils.set_random_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE
augmentador = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.03, fill_mode="nearest", seed=SEED),
    layers.RandomZoom(0.10, fill_mode="nearest", seed=SEED),
    layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", seed=SEED),
    layers.RandomContrast(0.10, seed=SEED),
], name="augmentation_final")


def cargar(ruta):
    img = tf.io.decode_png(tf.io.read_file(ruta), channels=1)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return tf.cast(img, tf.uint8)


def hacer_ds(df, entrenamiento=False):
    rutas = df["local_file"].astype(str).tolist()
    etiquetas = df["tumor_type"].map(CLASS_MAP).astype("float32").tolist()
    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    ds = ds.map(lambda p, y: (cargar(p), y), num_parallel_calls=AUTOTUNE)
    if entrenamiento:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)

    def adaptar(x, y):
        x = tf.cast(x, tf.float32)
        if entrenamiento:
            x = tf.clip_by_value(augmentador(x, training=True), 0.0, 255.0)
        x = tf.repeat(x, 3, axis=-1)                       # 1 canal -> 3 (réplica)
        x = keras.applications.resnet50.preprocess_input(x)
        return x, y

    return ds.map(adaptar, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# --- modelo -------------------------------------------------------------------
base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False
x = layers.GlobalAveragePooling2D(name="gap")(base.output)
x = layers.Dropout(0.4, name="dropout")(x)
x = layers.Dense(128, activation="relu", name="densa")(x)
x = layers.Dropout(0.3, name="dropout2")(x)
out = layers.Dense(3, activation="softmax", name="salida")(x)
model = keras.Model(base.input, out, name="resnet50_ft_cheng_final")
assert model.count_params() == EXPECTED_TOTAL_PARAMS
assert n_trainable(model) == EXPECTED_HEAD_TRAINABLE_PARAMS

ds_train = hacer_ds(train, entrenamiento=True)

print("FASE 1 — backbone congelado · 6 épocas · Adam 1e-4")
model.compile(keras.optimizers.Adam(LR_HEAD), loss="sparse_categorical_crossentropy",
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
hist_head = model.fit(ds_train, epochs=EPOCHS_HEAD, verbose=1)

HEAD = {"gap", "dropout", "densa", "dropout2", "salida"}
for layer in model.layers:
    layer.trainable = layer.name in HEAD
candidatas = [l for l in model.layers
              if l.name not in HEAD and not isinstance(l, layers.BatchNormalization)]
desbloqueadas = candidatas[-N_CAPAS_FINETUNE:]
for layer in desbloqueadas:
    layer.trainable = True
assert n_trainable(model) == EXPECTED_FT_TRAINABLE_PARAMS

print("FASE 2 — fine-tuning últimas 20 capas no-BN · 6 épocas · Adam 1e-5 · BN congelado")
model.compile(keras.optimizers.Adam(LR_FINETUNE), loss="sparse_categorical_crossentropy",
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
hist_ft = model.fit(ds_train, epochs=EPOCHS_FINETUNE, verbose=1)

model_keras = OUT / "resnet50_ft_cheng_final.keras"
model_h5 = OUT / "resnet50_ft_cheng_final.h5"
model.save(model_keras)
model.save(model_h5)                  # formato que carga cloud-run-backend/main.py
model_hash = sha256_file(model_keras)

# --- evaluación ÚNICA sobre test ---------------------------------------------
STARTED.write_text(json.dumps({"modelo_sha256": model_hash, "split_sha256": SPLIT_SHA256,
                               "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
prob = model.predict(hacer_ds(test), verbose=1)
sl = test[["patient_id", "tumor_type", "local_file", "slice_id"]].copy()
for i, c in enumerate(CLASSES):
    sl[f"p_{c}"] = prob[:, i]
sl.to_csv(OUT / "predicciones_slice_test.csv", index=False)

# agregación por paciente: media de probabilidades + argmax (Anexo H.1)
pp = (sl.groupby("patient_id")
        .agg(tumor_type=("tumor_type", "first"), n_cortes=("patient_id", "size"),
             **{f"p_{c}": (f"p_{c}", "mean") for c in CLASSES})
        .reset_index())
P = pp[[f"p_{c}" for c in CLASSES]].to_numpy()
y_true = pp["tumor_type"].map(CLASS_MAP).to_numpy()
y_pred = P.argmax(1)
pp["pred"] = [CLASSES[i] for i in y_pred]
pp["confianza"] = P.max(1)
pp.to_csv(OUT / "predicciones_paciente_test.csv", index=False)

rng = np.random.default_rng(SEED)
boot = []
for _ in range(2000):                      # bootstrap simple por paciente
    idx = rng.integers(0, len(y_true), len(y_true))
    if len(np.unique(y_true[idx])) == 3:
        boot.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))

res = {
    "protocolo": "C.1 (reimplementación, ver docstring)",
    "n_pacientes_test": int(len(pp)), "n_cortes_test": int(len(sl)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    "recall": dict(zip(CLASSES, map(float, recall_score(y_true, y_pred, labels=[0, 1, 2], average=None)))),
    "precision": dict(zip(CLASSES, map(float, precision_score(y_true, y_pred, labels=[0, 1, 2],
                                                               average=None, zero_division=0)))),
    "matriz_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    "ic95_bootstrap_balanced_accuracy": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "modelo_sha256": model_hash, "split_sha256": SPLIT_SHA256, "seed": SEED,
    "historial_head": {k: [float(v) for v in vs] for k, vs in hist_head.history.items()},
    "historial_finetune": {k: [float(v) for v in vs] for k, vs in hist_ft.history.items()},
}
(OUT / "resultado_evaluacion_final.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
COMPLETED.write_text(json.dumps({"modelo_sha256": model_hash, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))

print(json.dumps({k: res[k] for k in ("balanced_accuracy", "accuracy", "macro_f1",
                                      "matriz_confusion", "ic95_bootstrap_balanced_accuracy")}, indent=2))
print("Referencia TFG (job 70930): BA 0.9389 · Acc 0.9429 · Macro-F1 0.9398")
