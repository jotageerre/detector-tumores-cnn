"""
construir_manifest_cheng.py
===========================

Herramienta AÑADIDA para el repositorio público (no forma parte de las
ejecuciones originales del TFG). Sirve para que cualquier persona pueda
preparar los datos de la tarea multiclase SIN tener que ejecutar el notebook
completo (que además descarga IXI, solo necesario para la tarea binaria
histórica).

Qué hace:
  1. Descarga los 4 ZIP del dataset de Cheng et al. desde figshare
     (DOI 10.6084/m9.figshare.1512427, licencia CC BY 4.0) y verifica su MD5.
  2. Lee cada .mat (cjdata.PID, cjdata.label, cjdata.image).
  3. Exporta cada corte a PNG 8 bits 256x256 con EXACTAMENTE la misma función
     `_a_png` del notebook original (recorte a la caja del cerebro +
     normalización por percentiles 1-99).
  4. Asigna a cada paciente el subconjunto (train/val/test) del split
     CONGELADO del TFG (data/splits/split_multiclase_cheng.csv). No genera
     ningún split nuevo.
  5. Comprueba que el split resultante reproduce la huella documentada
     (fda7e2daeb9ec2de...) y escribe <root>/results/artifacts/manifest_final.csv,
     que es el fichero que esperan los scripts de training/robustez_5cv/ y
     training/entrenar_modelo_final.py.

El manifiesto generado solo contiene las filas de Cheng (3.064 cortes), por lo
que su SHA-256 de fichero NO coincidirá con el manifest_final.csv original
(que tenía 5.872 filas, incluía IXI y rutas del clúster). Lo que sí debe
coincidir es la huella del split por paciente, que el script verifica.

Uso:
    python construir_manifest_cheng.py --root ./tfg_run \
        --split ../../data/splits/split_multiclase_cheng.csv
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------
# Constantes copiadas literalmente del notebook 02 (Sección 1)
# --------------------------------------------------------------------------
FIGSHARE_ARTICLE_ID = 1512427
FIGSHARE_ZIPS = {  # nombre -> (file_id, md5 publicado por figshare)
    "brainTumorDataPublic_1-766.zip": (3381290, "74b949ad33f042e6e103523091cd1428"),
    "brainTumorDataPublic_767-1532.zip": (3381296, "7e8a875500d2c8a346f270538e29890e"),
    "brainTumorDataPublic_1533-2298.zip": (3381293, "8227bf6080cb71f15a88be8d25c79ae7"),
    "brainTumorDataPublic_2299-3064.zip": (3381302, "b378a80d6174e5317d59eb28430c6652"),
}
FIGSHARE_DOWNLOAD = "https://ndownloader.figshare.com/files/{file_id}"
CHENG_URL = f"https://doi.org/10.6084/m9.figshare.{FIGSHARE_ARTICLE_ID}"
CHENG_TIPO = {1: "meningioma", 2: "glioma", 3: "pituitario"}
PNG_SIZE = (256, 256)

SPLIT_SHA256_ESPERADO = "fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092"


def md5_de(path: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_de(path: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bloque), b""):
            h.update(chunk)
    return h.hexdigest()


def descargar(url: str, destino: Path, md5_esperado: str) -> Path:
    if destino.exists() and md5_de(destino) == md5_esperado:
        print(f"  [ok] {destino.name} ya presente y verificado")
        return destino
    print(f"  descargando {destino.name} ...")
    tmp = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(destino)
    real = md5_de(destino)
    if real != md5_esperado:
        raise RuntimeError(f"MD5 incorrecto en {destino.name}: {real} != {md5_esperado}")
    print(f"  [ok] {destino.name} descargado y MD5 verificado")
    return destino


# --------------------------------------------------------------------------
# Funciones copiadas literalmente del notebook 02 (Sección 1.3)
# --------------------------------------------------------------------------
def _a_png(arr2d, destino):
    """Recorta a la caja del cerebro, normaliza por percentiles y guarda PNG 8-bit."""
    a = np.nan_to_num(np.asarray(arr2d, dtype=np.float32))
    if a.ndim != 2 or min(a.shape) < 8:
        return None
    # --- caja del cerebro: todo lo que supere el 10 % del rango robusto ---
    p1, p99 = np.percentile(a, (1, 99))
    if p99 - p1 < 1e-6:
        return None
    norm = np.clip((a - p1) / (p99 - p1), 0, 1)
    mask = norm > 0.10
    if mask.sum() < 64:
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # caja cuadrada centrada con un 4 % de margen, para no deformar la anatomía
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    lado = max(y1 - y0, x1 - x0) * 1.04
    y0 = int(max(0, round(cy - lado / 2))); y1 = int(min(a.shape[0], round(cy + lado / 2)))
    x0 = int(max(0, round(cx - lado / 2))); x1 = int(min(a.shape[1], round(cx + lado / 2)))
    recorte = norm[y0:y1, x0:x1]
    if min(recorte.shape) < 8:
        return None
    im = Image.fromarray((recorte * 255).astype(np.uint8), mode="L")
    im = im.resize(PNG_SIZE, Image.LANCZOS)
    im.save(destino, format="PNG", optimize=True)
    return im


def _leer_cjdata(path):
    """Devuelve (PID, label, image) de un .mat de Cheng.
    Los .mat originales son MATLAB v7.3 (HDF5); se admite también v7 por si
    alguien redistribuye la colección en el formato antiguo."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            g = f["cjdata"]
            pid_raw = np.array(g["PID"]).flatten()
            pid = "".join(chr(int(c)) for c in pid_raw if int(c) > 0).strip()
            label = int(np.array(g["label"]).flatten()[0])
            img = np.array(g["image"]).T          # HDF5 devuelve la matriz transpuesta
            return pid, label, img
    except OSError:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)["cjdata"]
        pid = str(m.PID).strip()
        return pid, int(m.label), np.asarray(m.image)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="tfg_run", help="Carpeta de trabajo (equivale a PROJECT_ROOT del notebook)")
    ap.add_argument("--split", required=True, help="Ruta a data/splits/split_multiclase_cheng.csv")
    ap.add_argument("--sin-descarga", action="store_true",
                    help="No descargar: usar los .mat ya presentes en <root>/data/raw/cheng_mat")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw_dir = root / "data" / "raw"
    cheng_dir = raw_dir / "cheng_mat"
    img_dir = root / "data" / "images"
    art_dir = root / "results" / "artifacts"
    for d in (raw_dir, cheng_dir, img_dir, art_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Descarga + extracción ------------------------------------------------
    if not args.sin_descarga and len(list(cheng_dir.glob("*.mat"))) < 3064:
        print("(A) Cheng et al. 2017 — figshare 10.6084/m9.figshare.1512427 (CC BY 4.0)")
        for nombre, (fid, md5) in FIGSHARE_ZIPS.items():
            z = descargar(FIGSHARE_DOWNLOAD.format(file_id=fid), raw_dir / nombre, md5)
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if m.lower().endswith(".mat"):
                        destino = cheng_dir / Path(m).name
                        if not destino.exists():
                            with zf.open(m) as src, open(destino, "wb") as dst:
                                shutil.copyfileobj(src, dst)
    ficheros = sorted(cheng_dir.glob("*.mat"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    print(f"  .mat disponibles: {len(ficheros)}")

    # 2) Split congelado ------------------------------------------------------
    split = pd.read_csv(args.split)
    asignacion = dict(zip(split["patient_id"], split["subset"]))

    # 3) Exportación a PNG + manifiesto ---------------------------------------
    filas = []
    for p in ficheros:
        pid, label, img = _leer_cjdata(p)
        destino = img_dir / f"cheng_{p.stem}.png"
        if _a_png(img, destino) is None:
            print(f"  [aviso] corte descartado por _a_png: {p.name}")
            continue
        patient_id = f"CHENG-{pid}"
        if patient_id not in asignacion:
            raise RuntimeError(f"{patient_id} ({p.name}) no está en el split congelado")
        w, h = Image.open(destino).size
        filas.append({
            "local_file": str(destino),
            "label": "tumor",
            "patient_id": patient_id,
            "patient_id_source": "real:cjdata.PID",
            "volume_id": "NO_DISPONIBLE",
            "slice_id": p.stem,
            "view": "NO_DISPONIBLE",
            "source_dataset": "figshare/Cheng2017",
            "source_url": CHENG_URL,
            "original_file": p.name,
            "site": "Nanfang/Tianjin (CN)",
            "tumor_type": CHENG_TIPO.get(label, f"desconocido({label})"),
            "modality": "T1-CE",
            "sha256": sha256_de(destino),
            "width": w,
            "height": h,
            "subset": asignacion[patient_id],
        })
    df = pd.DataFrame(filas)

    # 4) Verificaciones -------------------------------------------------------
    assert len(df) == 3064, f"se esperaban 3064 cortes y hay {len(df)}"
    assert df["patient_id"].nunique() == 233, df["patient_id"].nunique()
    assert (df.groupby("patient_id")["tumor_type"].nunique() == 1).all()
    assert df["sha256"].is_unique, "hay PNG duplicados exactos"

    pac = (df.groupby("patient_id")
             .agg(subset=("subset", "first"), tumor_type=("tumor_type", "first"),
                  n_slices=("patient_id", "size"))
             .reset_index().sort_values("patient_id").reset_index(drop=True))
    huella = hashlib.sha256(pac[["patient_id", "subset", "tumor_type"]]
                            .to_csv(index=False).encode("utf-8")).hexdigest()
    print(f"  huella del split : {huella}")
    if huella != SPLIT_SHA256_ESPERADO:
        raise RuntimeError("La huella del split NO coincide con la documentada en el TFG (Anexo H.3).")
    esperado = split.set_index("patient_id")["n_slices"]
    obtenido = pac.set_index("patient_id")["n_slices"]
    assert (esperado.sort_index() == obtenido.sort_index()).all(), "nº de cortes por paciente distinto"

    salida = art_dir / "manifest_final.csv"
    df.to_csv(salida, index=False)
    print(f"\nOK: {salida}  ({len(df)} cortes, 233 pacientes, split verificado)")
    print(df.groupby(["subset", "tumor_type"])["patient_id"].nunique().unstack())
    return 0


if __name__ == "__main__":
    sys.exit(main())
