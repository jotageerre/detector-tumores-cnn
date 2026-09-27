import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path("/shared/home/FYK3492/TFG_tumores/tfg_run")

MANIFEST = ROOT / "results/artifacts/manifest_final.csv"
OUT = ROOT / "results/tumor_type/robustez_5cv/splits"

OUT.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 20260904

CLASSES = ["glioma", "meningioma", "pituitario"]

ORIGINAL_SPLIT_SHA256 = (
    "fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


data = pd.read_csv(MANIFEST)

cheng = data[
    data["source_dataset"] == "figshare/Cheng2017"
].reset_index(drop=True)

assert len(cheng) == 3064
assert cheng["patient_id"].nunique() == 233

assert (
    cheng.groupby("patient_id")["tumor_type"].nunique() == 1
).all()

assert (
    cheng.groupby("patient_id")["subset"].nunique() == 1
).all()


# ============================================================================
# Comprobación de que seguimos trabajando sobre el split histórico esperado
# ============================================================================

pac_original = (
    cheng.groupby("patient_id")
    .agg(
        subset=("subset", "first"),
        tumor_type=("tumor_type", "first"),
        n_slices=("patient_id", "size"),
    )
    .reset_index()
    .sort_values("patient_id")
    .reset_index(drop=True)
)

split_txt = pac_original[
    ["patient_id", "subset", "tumor_type"]
].to_csv(index=False)

original_hash = hashlib.sha256(
    split_txt.encode("utf-8")
).hexdigest()

assert original_hash == ORIGINAL_SPLIT_SHA256, (
    original_hash,
    ORIGINAL_SPLIT_SHA256,
)


# ============================================================================
# Tabla patient-level usada para la NUEVA 5CV
# ============================================================================

patients = pac_original[
    ["patient_id", "tumor_type", "n_slices"]
].copy()

patients = patients.sort_values(
    "patient_id"
).reset_index(drop=True)

assert patients["patient_id"].nunique() == 233

counts = patients["tumor_type"].value_counts().to_dict()

assert counts == {
    "glioma": 89,
    "meningioma": 82,
    "pituitario": 62,
}, counts


sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SPLIT_SEED,
)

X = np.zeros((len(patients), 1), dtype=np.float32)
y = patients["tumor_type"].to_numpy()
groups = patients["patient_id"].to_numpy()

filas_resumen = []
fold_records = []
todos_test = []

print("=" * 100)
print("NUEVO PARTICIONADO 5-FOLD PATIENT-LEVEL")
print("=" * 100)
print("Pacientes:", len(patients))
print("Clases:", counts)
print("Split seed:", SPLIT_SEED)
print("Split histórico verificado:", original_hash)
print()

for fold_zero, (itr, ite) in enumerate(
    sgkf.split(X, y, groups=groups)
):
    fold = fold_zero + 1

    train_pat = patients.iloc[itr].copy()
    test_pat = patients.iloc[ite].copy()

    train_ids = sorted(
        train_pat["patient_id"].astype(str).tolist()
    )

    test_ids = sorted(
        test_pat["patient_id"].astype(str).tolist()
    )

    assert not (set(train_ids) & set(test_ids))
    assert len(train_ids) + len(test_ids) == 233

    train_counts = {
        c: int((train_pat["tumor_type"] == c).sum())
        for c in CLASSES
    }

    test_counts = {
        c: int((test_pat["tumor_type"] == c).sum())
        for c in CLASSES
    }

    payload = {
        "fold": fold,
        "fold_index": fold_zero,
        "metodo": "StratifiedGroupKFold patient-level",
        "n_splits": 5,
        "split_seed": SPLIT_SEED,
        "groups": "patient_id",
        "stratify": "tumor_type a nivel de paciente",
        "n_train_patients": len(train_ids),
        "n_test_patients": len(test_ids),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "train_patient_ids": train_ids,
        "test_patient_ids": test_ids,
        "original_frozen_split_sha256": original_hash,
    }

    fold_path = OUT / f"fold_{fold}.json"

    fold_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    fold_hash = sha256_file(fold_path)

    (OUT / f"fold_{fold}.sha256").write_text(
        f"{fold_hash}  {fold_path.name}\n",
        encoding="utf-8",
    )

    fold_records.append({
        "fold": fold,
        "json": str(fold_path.relative_to(ROOT)),
        "sha256": fold_hash,
        "n_train_patients": len(train_ids),
        "n_test_patients": len(test_ids),
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
    })

    todos_test.extend(test_ids)

    filas_resumen.append({
        "fold": fold,
        "train_n": len(train_ids),
        "train_glioma": train_counts["glioma"],
        "train_meningioma": train_counts["meningioma"],
        "train_pituitario": train_counts["pituitario"],
        "test_n": len(test_ids),
        "test_glioma": test_counts["glioma"],
        "test_meningioma": test_counts["meningioma"],
        "test_pituitario": test_counts["pituitario"],
    })


# ============================================================================
# Cada paciente debe aparecer exactamente una vez como test
# ============================================================================

assert len(todos_test) == 233
assert len(set(todos_test)) == 233
assert set(todos_test) == set(patients["patient_id"].astype(str))

resumen = pd.DataFrame(filas_resumen)

resumen.to_csv(
    OUT / "distribucion_folds.csv",
    index=False,
)

semantic_payload = {
    "split_seed": SPLIT_SEED,
    "folds": [
        {
            "fold": r["fold"],
            "test_patient_ids": json.loads(
                (ROOT / r["json"]).read_text(encoding="utf-8")
            )["test_patient_ids"],
        }
        for r in fold_records
    ],
}

cv_semantic_sha256 = hashlib.sha256(
    json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

master = {
    "tipo": "5-fold robustness validation split",
    "n_patients": 233,
    "classes": CLASSES,
    "class_counts": counts,
    "n_splits": 5,
    "split_seed": SPLIT_SEED,
    "groups": "patient_id",
    "original_frozen_split_sha256": original_hash,
    "cv_semantic_sha256": cv_semantic_sha256,
    "split_script": str(Path(__file__).name),
    "split_script_sha256": sha256_file(Path(__file__)),
    "folds": fold_records,
}

master_path = OUT / "folds_master.json"

master_path.write_text(
    json.dumps(
        master,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ),
    encoding="utf-8",
)

master_hash = sha256_file(master_path)

(OUT / "folds_master.sha256").write_text(
    f"{master_hash}  {master_path.name}\n",
    encoding="utf-8",
)

print(resumen.to_string(index=False))

print()
print("Todos los pacientes aparecen exactamente una vez como TEST: OK")
print("CV semantic SHA256:", cv_semantic_sha256)
print("Master SHA256:", master_hash)
print("Split script SHA256:", master["split_script_sha256"])
print()
print("Guardado en:", OUT)
