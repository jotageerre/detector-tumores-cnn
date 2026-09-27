import argparse
import gc
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
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path("/shared/home/FYK3492/TFG_tumores/tfg_run")

MANIFEST = ROOT / "results/artifacts/manifest_final.csv"
PROTOCOL = ROOT / "frozen/robustez_5cv/protocolo_5cv_robustez.md"

BASE = ROOT / "results/tumor_type/robustez_5cv"
SPLITS = BASE / "splits"
MODELS = BASE / "models"
METRICS = BASE / "metrics"
PREDICTIONS = BASE / "predictions"

for d in (MODELS, METRICS, PREDICTIONS):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["glioma", "meningioma", "pituitario"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_HEAD = 6
EPOCHS_FINETUNE = 6

LR_HEAD = 1e-4
LR_FINETUNE = 1e-5

N_CAPAS_FINETUNE = 20

EXPECTED_TOTAL_PARAMS = 23850371
EXPECTED_HEAD_TRAINABLE_PARAMS = 262659
EXPECTED_FT_TRAINABLE_PARAMS = 14691331


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def n_trainable(model):
    return int(
        sum(
            np.prod(w.shape)
            for w in model.trainable_weights
        )
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--fold-index",
    required=True,
    type=int,
    choices=range(5),
)
args = parser.parse_args()

fold_index = args.fold_index
fold = fold_index + 1
seed = 1000 + fold

FOLD_JSON = SPLITS / f"fold_{fold}.json"
MASTER_JSON = SPLITS / "folds_master.json"

MODEL_PATH = MODELS / f"resnet50_robustez5cv_fold_{fold}.keras"
RESULT_PATH = METRICS / f"resultado_5cv_fold_{fold}.json"
PATIENT_PRED = PREDICTIONS / f"predicciones_paciente_5cv_fold_{fold}.csv"
SLICE_PRED = PREDICTIONS / f"predicciones_slice_5cv_fold_{fold}.csv"
CM_PATH = METRICS / f"confusion_matrix_5cv_fold_{fold}.csv"

for p in (MODEL_PATH, RESULT_PATH, PATIENT_PRED):
    if p.exists():
        raise RuntimeError(
            f"ABORTADO: ya existe {p}. "
            "No se sobrescribe un fold ya ejecutado."
        )

master = json.loads(
    MASTER_JSON.read_text(encoding="utf-8")
)

fold_meta = next(
    x for x in master["folds"]
    if x["fold"] == fold
)

if sha256_file(FOLD_JSON) != fold_meta["sha256"]:
    raise RuntimeError(
        "El JSON del fold ya no coincide con el hash congelado."
    )

fd = json.loads(FOLD_JSON.read_text(encoding="utf-8"))

train_ids = set(map(str, fd["train_patient_ids"]))
test_ids = set(map(str, fd["test_patient_ids"]))

assert train_ids.isdisjoint(test_ids)
assert len(train_ids | test_ids) == 233

print("=" * 80)
print(f"ROBUSTEZ 5CV — FOLD {fold}/5")
print("=" * 80)
print("Seed entrenamiento:", seed)
print("Fold SHA256:", sha256_file(FOLD_JSON))
print("CV semantic SHA256:", master["cv_semantic_sha256"])
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

data = pd.read_csv(MANIFEST)

cheng = data[
    data["source_dataset"] == "figshare/Cheng2017"
].reset_index(drop=True)

assert len(cheng) == 3064
assert cheng["patient_id"].nunique() == 233

train = cheng[
    cheng["patient_id"].astype(str).isin(train_ids)
].reset_index(drop=True)

test = cheng[
    cheng["patient_id"].astype(str).isin(test_ids)
].reset_index(drop=True)

assert train["patient_id"].nunique() == len(train_ids)
assert test["patient_id"].nunique() == len(test_ids)

assert set(train["patient_id"].astype(str)).isdisjoint(
    set(test["patient_id"].astype(str))
)

train_counts = (
    train.groupby("patient_id")["tumor_type"]
    .first()
    .value_counts()
    .to_dict()
)

test_counts = (
    test.groupby("patient_id")["tumor_type"]
    .first()
    .value_counts()
    .to_dict()
)

assert train_counts == fd["train_class_counts"]
assert test_counts == fd["test_class_counts"]

print(
    f"TRAIN: {len(train_ids)} pacientes / {len(train)} slices / {train_counts}"
)
print(
    f"EVAL : {len(test_ids)} pacientes / {len(test)} slices / {test_counts}"
)


keras.utils.set_random_seed(seed)
AUTOTUNE = tf.data.AUTOTUNE

augmentador = keras.Sequential(
    [
        keras.layers.RandomFlip(
            "horizontal",
            seed=seed,
        ),
        keras.layers.RandomRotation(
            0.03,
            fill_mode="nearest",
            seed=seed,
        ),
        keras.layers.RandomZoom(
            0.10,
            fill_mode="nearest",
            seed=seed,
        ),
        keras.layers.RandomTranslation(
            0.05,
            0.05,
            fill_mode="nearest",
            seed=seed,
        ),
        keras.layers.RandomContrast(
            0.10,
            seed=seed,
        ),
    ],
    name=f"augmentation_fold_{fold}",
)


def cargar(ruta):
    img = tf.io.decode_png(
        tf.io.read_file(ruta),
        channels=1,
    )

    img = tf.image.resize(
        img,
        IMG_SIZE,
        method="bilinear",
    )

    return tf.cast(img, tf.uint8)


def hacer_ds(df, entrenamiento=False):
    rutas = df["local_file"].astype(str).tolist()

    etiquetas = (
        df["tumor_type"]
        .map(CLASS_MAP)
        .astype("float32")
        .tolist()
    )

    @tf.autograph.experimental.do_not_convert
    def leer(p, y):
        return cargar(p), y

    ds = tf.data.Dataset.from_tensor_slices(
        (rutas, etiquetas)
    )

    ds = ds.map(
        leer,
        num_parallel_calls=AUTOTUNE,
    )

    if entrenamiento:
        ds = ds.shuffle(
            min(len(df), 4096),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(BATCH_SIZE)

    @tf.autograph.experimental.do_not_convert
    def adaptar(x, y):
        x = tf.cast(x, tf.float32)

        if entrenamiento:
            x = augmentador(
                x,
                training=True,
            )
            x = tf.clip_by_value(
                x,
                0.0,
                255.0,
            )

        x = tf.repeat(
            x,
            3,
            axis=-1,
        )

        x = keras.applications.resnet50.preprocess_input(x)

        return x, y

    ds = ds.map(
        adaptar,
        num_parallel_calls=AUTOTUNE,
    )

    return ds.prefetch(AUTOTUNE)


keras.backend.clear_session()
keras.utils.set_random_seed(seed)

base = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

base.trainable = False

x = layers.GlobalAveragePooling2D(
    name="gap"
)(base.output)

x = layers.Dropout(
    0.4,
    name="dropout",
)(x)

x = layers.Dense(
    128,
    activation="relu",
    name="densa",
)(x)

x = layers.Dropout(
    0.3,
    name="dropout2",
)(x)

out = layers.Dense(
    3,
    activation="softmax",
    name="salida",
)(x)

model = keras.Model(
    base.input,
    out,
    name=f"resnet50_robustez5cv_fold_{fold}",
)

assert model.count_params() == EXPECTED_TOTAL_PARAMS
assert n_trainable(model) == EXPECTED_HEAD_TRAINABLE_PARAMS

model.compile(
    optimizer=keras.optimizers.Adam(LR_HEAD),
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ],
)

ds_train = hacer_ds(
    train,
    entrenamiento=True,
)

print()
print("=" * 80)
print("FASE 1 — BACKBONE CONGELADO")
print("6 épocas fijas · Adam 1e-4")
print("=" * 80)

t0 = time.time()

hist_head = model.fit(
    ds_train,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

seg_head = time.time() - t0


HEAD_NAMES = {
    "gap",
    "dropout",
    "densa",
    "dropout2",
    "salida",
}

for layer in model.layers:
    layer.trainable = layer.name in HEAD_NAMES

candidatas = [
    layer
    for layer in model.layers
    if layer.name not in HEAD_NAMES
    and not isinstance(
        layer,
        keras.layers.BatchNormalization,
    )
]

desbloqueadas = candidatas[
    -N_CAPAS_FINETUNE:
]

assert len(desbloqueadas) == 20

for layer in desbloqueadas:
    layer.trainable = True

assert all(
    not isinstance(
        layer,
        keras.layers.BatchNormalization,
    )
    for layer in desbloqueadas
)

assert n_trainable(model) == EXPECTED_FT_TRAINABLE_PARAMS

model.compile(
    optimizer=keras.optimizers.Adam(
        LR_FINETUNE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ],
)

print()
print("=" * 80)
print("FASE 2 — FINE-TUNING LIMITADO")
print("6 épocas fijas · Adam 1e-5")
print("BatchNormalization congelado")
print("=" * 80)

for layer in desbloqueadas:
    print(" ", layer.name)

t0 = time.time()

hist_ft = model.fit(
    ds_train,
    epochs=EPOCHS_FINETUNE,
    verbose=1,
)

seg_ft = time.time() - t0

model.save(MODEL_PATH)
model_hash = sha256_file(MODEL_PATH)

print()
print("Modelo guardado antes de evaluar el fold.")
print("SHA256:", model_hash)


ds_eval = hacer_ds(
    test,
    entrenamiento=False,
)

prob_slice = model.predict(
    ds_eval,
    verbose=1,
)

assert prob_slice.shape == (
    len(test),
    3,
)

slice_df = test[
    [
        "patient_id",
        "tumor_type",
        "local_file",
        "slice_id",
    ]
].copy()

for i, clase in enumerate(CLASSES):
    slice_df[f"p_{clase}"] = prob_slice[:, i]

slice_df.to_csv(
    SLICE_PRED,
    index=False,
)

patient_df = (
    slice_df.groupby("patient_id")
    .agg(
        tumor_type=("tumor_type", "first"),
        n_cortes=("patient_id", "size"),
        p_glioma=("p_glioma", "mean"),
        p_meningioma=("p_meningioma", "mean"),
        p_pituitario=("p_pituitario", "mean"),
    )
    .reset_index()
)

assert len(patient_df) == len(test_ids)

P = patient_df[
    [
        "p_glioma",
        "p_meningioma",
        "p_pituitario",
    ]
].to_numpy()

y_true = (
    patient_df["tumor_type"]
    .map(CLASS_MAP)
    .astype(int)
    .to_numpy()
)

y_pred = P.argmax(axis=1)

patient_df["pred_idx"] = y_pred
patient_df["pred"] = [
    CLASSES[i]
    for i in y_pred
]
patient_df["correcto"] = (
    patient_df["pred"]
    == patient_df["tumor_type"]
)
patient_df["confianza"] = P.max(axis=1)

patient_df.to_csv(
    PATIENT_PRED,
    index=False,
)

ba = balanced_accuracy_score(
    y_true,
    y_pred,
)

acc = accuracy_score(
    y_true,
    y_pred,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
)

rec = recall_score(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    average=None,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

pd.DataFrame(
    cm,
    index=[
        f"real_{c}"
        for c in CLASSES
    ],
    columns=[
        f"pred_{c}"
        for c in CLASSES
    ],
).to_csv(CM_PATH)

resultado = {
    "tipo_experimento":
        "5-fold patient-level robustness validation",
    "modelo":
        "ResNet50 fine-tuned previamente seleccionado",
    "fold": fold,
    "fold_index": fold_index,
    "seed": seed,
    "n_train_patients": len(train_ids),
    "n_eval_patients": len(test_ids),
    "n_train_slices": len(train),
    "n_eval_slices": len(test),
    "balanced_accuracy_paciente": float(ba),
    "accuracy_paciente": float(acc),
    "macro_f1_paciente": float(macro_f1),
    "recall_por_clase": {
        c: float(v)
        for c, v in zip(CLASSES, rec)
    },
    "matriz_confusion": cm.tolist(),
    "epochs_head": EPOCHS_HEAD,
    "epochs_finetune": EPOCHS_FINETUNE,
    "lr_head": LR_HEAD,
    "lr_finetune": LR_FINETUNE,
    "n_capas_finetune": N_CAPAS_FINETUNE,
    "batch_norm_congelado": True,
    "capas_desbloqueadas": [
        layer.name
        for layer in desbloqueadas
    ],
    "parametros": int(model.count_params()),
    "parametros_entrenables": n_trainable(model),
    "segundos_head": round(seg_head, 2),
    "segundos_finetune": round(seg_ft, 2),
    "modelo_path": str(
        MODEL_PATH.relative_to(ROOT)
    ),
    "modelo_sha256": model_hash,
    "fold_json": str(
        FOLD_JSON.relative_to(ROOT)
    ),
    "fold_json_sha256": sha256_file(FOLD_JSON),
    "folds_master_sha256": sha256_file(MASTER_JSON),
    "cv_semantic_sha256": master["cv_semantic_sha256"],
    "manifest_sha256": sha256_file(MANIFEST),
    "protocol_sha256": sha256_file(PROTOCOL),
    "training_script_sha256": sha256_file(
        Path(__file__)
    ),
    "historial_head": {
        k: [float(x) for x in v]
        for k, v in hist_head.history.items()
    },
    "historial_finetune": {
        k: [float(x) for x in v]
        for k, v in hist_ft.history.items()
    },
    "fecha": time.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
}

RESULT_PATH.write_text(
    json.dumps(
        resultado,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 80)
print(f"FOLD {fold} FINALIZADO")
print("=" * 80)
print(f"Balanced accuracy : {ba:.6f}")
print(f"Accuracy          : {acc:.6f}")
print(f"Macro-F1          : {macro_f1:.6f}")
print("Recall:", resultado["recall_por_clase"])
print("Matriz:")
print(cm)
print("Resultado:", RESULT_PATH)

del model
keras.backend.clear_session()
gc.collect()
