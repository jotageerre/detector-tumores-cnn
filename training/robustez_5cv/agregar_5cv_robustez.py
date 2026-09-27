import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path("/shared/home/FYK3492/TFG_tumores/tfg_run")

BASE = ROOT / "results/tumor_type/robustez_5cv"
SPLITS = BASE / "splits"
MODELS = BASE / "models"
METRICS = BASE / "metrics"
PREDICTIONS = BASE / "predictions"
DIAGNOSTICS = ROOT / "results/diagnostics/robustez_5cv"

PROTOCOL = ROOT / "frozen/robustez_5cv/protocolo_5cv_robustez.md"

OUT_JSON = METRICS / "resultado_5cv_validacion_robustez.json"
OUT_FOLDS = BASE / "resumen_5cv_folds.csv"
OUT_PATIENTS = PREDICTIONS / "predicciones_paciente_5cv_oof_233.csv"
OUT_CM = METRICS / "confusion_matrix_5cv_agregada.csv"
OUT_HASHES = DIAGNOSTICS / "sha256_resultados_5cv.txt"
LOCK = BASE / "VALIDACION_5CV_COMPLETADA.lock"

CLASSES = ["glioma", "meningioma", "pituitario"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


master_path = SPLITS / "folds_master.json"

master = json.loads(
    master_path.read_text(encoding="utf-8")
)

assert master["n_patients"] == 233
assert master["n_splits"] == 5
assert master["groups"] == "patient_id"

fold_rows = []
patient_frames = []
cms = []
fold_results = []

for fold in range(1, 6):

    result_path = METRICS / f"resultado_5cv_fold_{fold}.json"
    pred_path = PREDICTIONS / f"predicciones_paciente_5cv_fold_{fold}.csv"
    model_path = MODELS / f"resnet50_robustez5cv_fold_{fold}.keras"
    fold_json = SPLITS / f"fold_{fold}.json"

    for p in (result_path, pred_path, model_path, fold_json):
        if not p.exists():
            raise FileNotFoundError(p)

    r = json.loads(
        result_path.read_text(encoding="utf-8")
    )

    assert r["fold"] == fold

    model_hash_real = sha256_file(model_path)

    assert model_hash_real == r["modelo_sha256"], (
        fold,
        model_hash_real,
        r["modelo_sha256"],
    )

    fold_meta = next(
        x for x in master["folds"]
        if x["fold"] == fold
    )

    assert sha256_file(fold_json) == fold_meta["sha256"]
    assert r["fold_json_sha256"] == fold_meta["sha256"]
    assert r["cv_semantic_sha256"] == master["cv_semantic_sha256"]

    p = pd.read_csv(pred_path)

    assert len(p) == r["n_eval_patients"]
    assert p["patient_id"].nunique() == len(p)

    p["fold"] = fold
    p["training_seed"] = r["seed"]

    patient_frames.append(p)

    cm = np.asarray(
        r["matriz_confusion"],
        dtype=int,
    )

    assert cm.shape == (3, 3)
    assert int(cm.sum()) == r["n_eval_patients"]

    cms.append(cm)
    fold_results.append(r)

    fold_rows.append({
        "fold": fold,
        "seed": r["seed"],
        "n_train_patients": r["n_train_patients"],
        "n_eval_patients": r["n_eval_patients"],
        "n_train_slices": r["n_train_slices"],
        "n_eval_slices": r["n_eval_slices"],
        "balanced_accuracy": r["balanced_accuracy_paciente"],
        "accuracy": r["accuracy_paciente"],
        "macro_f1": r["macro_f1_paciente"],
        "recall_glioma": r["recall_por_clase"]["glioma"],
        "recall_meningioma": r["recall_por_clase"]["meningioma"],
        "recall_pituitario": r["recall_por_clase"]["pituitario"],
        "modelo_sha256": model_hash_real,
    })


fold_df = pd.DataFrame(fold_rows)

assert len(fold_df) == 5

fold_df.to_csv(
    OUT_FOLDS,
    index=False,
)


all_patients = pd.concat(
    patient_frames,
    ignore_index=True,
)

assert len(all_patients) == 233
assert all_patients["patient_id"].nunique() == 233
assert all_patients["patient_id"].duplicated().sum() == 0

all_patients = all_patients.sort_values(
    "patient_id"
).reset_index(drop=True)

all_patients.to_csv(
    OUT_PATIENTS,
    index=False,
)


metricas_folds = {}

for col in (
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
):
    values = fold_df[col].to_numpy(dtype=float)

    metricas_folds[col] = {
        "media": float(np.mean(values)),
        "desviacion_tipica_muestral": float(
            np.std(values, ddof=1)
        ),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "valores": [
            float(v)
            for v in values
        ],
    }


y_true = (
    all_patients["tumor_type"]
    .map(CLASS_MAP)
    .astype(int)
    .to_numpy()
)

y_pred = (
    all_patients["pred"]
    .map(CLASS_MAP)
    .astype(int)
    .to_numpy()
)

assert not np.isnan(y_true).any()
assert not np.isnan(y_pred).any()


ba_global = balanced_accuracy_score(
    y_true,
    y_pred,
)

acc_global = accuracy_score(
    y_true,
    y_pred,
)

macro_f1_global = f1_score(
    y_true,
    y_pred,
    average="macro",
)

recall_global = recall_score(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    average=None,
    zero_division=0,
)

cm_global = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

cm_sumada = np.sum(
    np.stack(cms),
    axis=0,
)

assert np.array_equal(
    cm_global,
    cm_sumada,
)

assert int(cm_global.sum()) == 233

n_correctos = int(
    np.trace(cm_global)
)

pd.DataFrame(
    cm_global,
    index=[
        f"real_{c}"
        for c in CLASSES
    ],
    columns=[
        f"pred_{c}"
        for c in CLASSES
    ],
).to_csv(
    OUT_CM
)


fold_entries = []

for r in fold_results:

    fold = r["fold"]

    model_path = (
        ROOT / r["modelo_path"]
    )

    result_path = (
        METRICS /
        f"resultado_5cv_fold_{fold}.json"
    )

    pred_path = (
        PREDICTIONS /
        f"predicciones_paciente_5cv_fold_{fold}.csv"
    )

    fold_entries.append({
        "fold": fold,
        "seed": r["seed"],
        "n_train_patients": r["n_train_patients"],
        "n_eval_patients": r["n_eval_patients"],
        "balanced_accuracy_paciente": r["balanced_accuracy_paciente"],
        "accuracy_paciente": r["accuracy_paciente"],
        "macro_f1_paciente": r["macro_f1_paciente"],
        "recall_por_clase": r["recall_por_clase"],
        "matriz_confusion": r["matriz_confusion"],
        "modelo_path": r["modelo_path"],
        "modelo_sha256": sha256_file(model_path),
        "resultado_path": str(
            result_path.relative_to(ROOT)
        ),
        "resultado_sha256": sha256_file(result_path),
        "predicciones_paciente_path": str(
            pred_path.relative_to(ROOT)
        ),
        "predicciones_paciente_sha256": sha256_file(pred_path),
        "fold_json": r["fold_json"],
        "fold_json_sha256": r["fold_json_sha256"],
    })


resultado = {
    "tipo_experimento":
        "5-fold patient-level robustness validation of previously selected ResNet50-FT",

    "objetivo":
        "Estimación interna complementaria de robustez del modelo previamente seleccionado; no utilizada para selección de arquitectura ni optimización de hiperparámetros.",

    "cohorte": {
        "dataset": "Cheng et al.",
        "n_patients": 233,
        "n_slices": 3064,
        "class_counts_patients": {
            "glioma": 89,
            "meningioma": 82,
            "pituitario": 62,
        },
    },

    "cross_validation": {
        "metodo": "StratifiedGroupKFold",
        "n_splits": 5,
        "groups": "patient_id",
        "split_seed": master["split_seed"],
        "cv_semantic_sha256": master["cv_semantic_sha256"],
        "training_seeds": [
            1001,
            1002,
            1003,
            1004,
            1005,
        ],
    },

    "modelo_fijo": {
        "arquitectura": "ResNet50 ImageNet + cabeza clasificadora",
        "epochs_head": 6,
        "lr_head": 0.0001,
        "epochs_finetune": 6,
        "lr_finetune": 0.00001,
        "n_capas_finetune": 20,
        "batch_normalization_congelado": True,
    },

    "metricas_entre_folds": metricas_folds,

    "metricas_globales_oof_233_pacientes": {
        "balanced_accuracy_paciente": float(ba_global),
        "accuracy_paciente": float(acc_global),
        "macro_f1_paciente": float(macro_f1_global),
        "recall_por_clase": {
            c: float(v)
            for c, v in zip(
                CLASSES,
                recall_global,
            )
        },
        "matriz_confusion_agregada": cm_global.tolist(),
        "n_correctos": n_correctos,
        "n_total": 233,
    },

    "folds": fold_entries,

    "resultado_test_original_referencia": {
        "n_patients": 35,
        "balanced_accuracy_paciente": 0.9388888888888888,
        "accuracy_paciente": 0.9428571428571428,
        "macro_f1_paciente": 0.9398282340311326,
        "nota":
            "Resultado histórico del protocolo final original. Se conserva sin cambios y no es sustituido por la presente 5CV.",
    },

    "hashes": {
        "protocolo_5cv_sha256": sha256_file(PROTOCOL),
        "folds_master_sha256": sha256_file(master_path),
        "split_script_sha256": master["split_script_sha256"],
        "training_script_sha256": fold_results[0]["training_script_sha256"],
        "aggregation_script_sha256": sha256_file(Path(__file__)),
    },

    "interpretacion": {
        "uso_correcto":
            "Validación interna complementaria de robustez del modelo seleccionado.",
        "no_es":
            "Validación externa independiente.",
        "nota":
            "Los 233 pacientes, incluidos los 35 del TEST histórico, participan en esta nueva cross-validation. Cada paciente aparece exactamente una vez como evaluación OOF en la 5CV.",
    },
}

OUT_JSON.write_text(
    json.dumps(
        resultado,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


hash_files = [
    OUT_JSON,
    OUT_FOLDS,
    OUT_PATIENTS,
    OUT_CM,
]

with OUT_HASHES.open(
    "w",
    encoding="utf-8",
) as f:
    for p in hash_files:
        f.write(
            f"{sha256_file(p)}  {p.relative_to(ROOT)}\n"
        )

result_hash = sha256_file(
    OUT_JSON
)

LOCK.write_text(
    "VALIDACION_5CV_COMPLETADA\n"
    f"resultado={OUT_JSON.relative_to(ROOT)}\n"
    f"sha256={result_hash}\n",
    encoding="utf-8",
)


print("=" * 80)
print("VALIDACIÓN DE ROBUSTEZ 5CV — RESULTADO AGREGADO")
print("=" * 80)

print()
print(fold_df.to_string(index=False))

print()
print("MEDIA ± SD ENTRE LOS 5 FOLDS")
print(
    "Balanced accuracy:",
    f"{metricas_folds['balanced_accuracy']['media']:.6f}",
    "±",
    f"{metricas_folds['balanced_accuracy']['desviacion_tipica_muestral']:.6f}",
)
print(
    "Accuracy:",
    f"{metricas_folds['accuracy']['media']:.6f}",
    "±",
    f"{metricas_folds['accuracy']['desviacion_tipica_muestral']:.6f}",
)
print(
    "Macro-F1:",
    f"{metricas_folds['macro_f1']['media']:.6f}",
    "±",
    f"{metricas_folds['macro_f1']['desviacion_tipica_muestral']:.6f}",
)

print()
print("OOF POOLED — 233 PACIENTES")
print(
    "Balanced accuracy:",
    f"{ba_global:.6f}",
)
print(
    "Accuracy:",
    f"{acc_global:.6f}",
)
print(
    "Macro-F1:",
    f"{macro_f1_global:.6f}",
)
print(
    "Recall:",
    {
        c: float(v)
        for c, v in zip(
            CLASSES,
            recall_global,
        )
    },
)

print()
print("Matriz de confusión agregada:")
print(cm_global)

print()
print(
    f"Correctos: {n_correctos}/233"
)

print()
print("Resultado:")
print(OUT_JSON)

print("SHA256:")
print(result_hash)
