# Baseline multiclase (Cheng 3 clases) — split congelado y documentación

**Fecha de verificación:** 2026-08-31
**Fuente de verdad usada:** `manifest_final.csv` de la ejecución SLURM real (job 70436, 2026-08-27), NO regenerado.

Este documento congela el split, la configuración y el resultado que produjeron el
79,7 % de balanced accuracy patient-level en la tarea de control de 3 clases
(glioma / meningioma / pituitario), tal como pide el usuario antes de tocar
ninguna función del framework.

## 1. Procedencia y verificación de identidad del código

El notebook auditado y corregido en la fase anterior (`TFG_tumores_cerebrales_patient_level_1.ipynb`)
**no es el mismo fichero** que se ejecutó en el job 70436
(`notebooks/TFG_tumores_cerebrales_patient_level.ipynb`, sin sufijo `_1`, dentro del tar.gz).
Antes de asumir que el split es reproducible se compararon ambos ficheros celda a celda:

- Las celdas que construyen el manifiesto, el `split_group` (deduplicación) y el
  reparto por paciente (secciones 1.1–1.8) son **funcionalmente idénticas** entre
  ambas versiones. Las únicas diferencias son de redacción de comentarios/markdown
  (tono más informal en la versión `_1`) y la incorporación de `guardar_figura()`
  para exportar las figuras a PNG, algo que la versión ejecutada en el 70436 no
  tenía (por eso ese run no generó ningún `.png`).
- El **Apéndice A** (la celda de control de 3 clases) es idéntico entre ambas
  versiones salvo el cambio ya solicitado de `gc` → `patient_ids_cheng`, que no
  afecta a ningún valor.
- Los 5 errores de sintaxis corregidos en la fase anterior **no existían** en el
  notebook que realmente se ejecutó en el 70436: se introdujeron después, al
  añadir las llamadas a `guardar_figura()`. Es decir, las correcciones restauran
  el comportamiento que ya funcionaba, no cambian metodología.

**Conclusión:** el split y la lógica de Apéndice A del notebook `_1` corregido
reproducirán, si se ejecutan hoy sobre el mismo `manifest_final.csv`, exactamente
el mismo reparto de pacientes que produjo el 79,7 %.

## 2. Split congelado — distribución por PACIENTE

| Subset | Glioma | Meningioma | Pituitario | Total pacientes |
|---|---|---|---|---|
| train | 63 | 62 | 38 | 163 |
| val   | 14 | 10 | 11 | 35 |
| test  | 12 | 10 | 13 | 35 |
| **Total** | **89** | **82** | **62** | **233** |

## 3. Split congelado — distribución por SLICE/IMAGEN

| Subset | Glioma | Meningioma | Pituitario | Total imágenes |
|---|---|---|---|---|
| train | 1026 | 528 | 537 | 2091 |
| val   | 220  | 100 | 179 | 499 |
| test  | 180  | 80  | 214 | 474 |
| **Total** | **1426** | **708** | **930** | **3064** |

## 4. Patient IDs exactos por subset

Listado completo en `split_multiclase_cheng.csv` / `.json` (233 filas, una por paciente).
El subconjunto de **test** (el más crítico, 35 pacientes) es:

CHENG-100416(glioma), CHENG-100572(meningioma), CHENG-100639(meningioma), CHENG-101017(pituitario),
CHENG-102446(meningioma), CHENG-102714(meningioma), CHENG-102935(meningioma), CHENG-103671(pituitario),
CHENG-104558(pituitario), CHENG-105187(pituitario), CHENG-105475(pituitario), CHENG-105936(pituitario),
CHENG-108479(glioma), CHENG-109769(pituitario), CHENG-109898(pituitario), CHENG-111077(meningioma),
CHENG-111532(meningioma), CHENG-112027(glioma), CHENG-112252(meningioma), CHENG-114018(meningioma),
CHENG-114094(pituitario), CHENG-97416(pituitario), CHENG-97875(meningioma), CHENG-98874(pituitario),
CHENG-98889(glioma), CHENG-98992(pituitario), CHENG-98995(pituitario), CHENG-MR033389B(glioma),
CHENG-MR033420(glioma), CHENG-MR034694(glioma), CHENG-MR040240B(glioma), CHENG-MR047953C(glioma),
CHENG-MR048944(glioma), CHENG-MR049453B(glioma), CHENG-MR051586(glioma)

## 5. Resultado de todos los asserts

| Comprobación | Resultado |
|---|---|
| train ∩ val = ∅ | OK |
| train ∩ test = ∅ | OK |
| val ∩ test = ∅ | OK |
| Cada patient_id aparece en un único subset | OK (0 pacientes con >1 subset) |
| Cada patient_id tiene una única clase | OK (0 pacientes con >1 tumor_type) |
| Todos los slices de un paciente caen en el mismo subset | OK |
| Nº total de pacientes Cheng = 233 | OK |
| Las 3 clases están presentes en train, val y test | OK |
| Nº de pacientes de test coincide con `control_3clases.json` (35) | OK — **coincide en cantidad Y en identidad exacta** (se reconstruyó patient a patient desde el mismo `manifest_final.csv` que usó esa ejecución, no solo se comparó el número) |
| Sin pacientes duplicados en la tabla de asignación | OK |

## 6. Huellas / fingerprint

```
SPLIT_HASH_MULTICLASE (16 chars, mismo formato que _hash_texto() del notebook):
  fda7e2daeb9ec2de

SPLIT_HASH_MULTICLASE (SHA-256 completo):
  fda7e2daeb9ec2de596e807728454329cd140398dd10f834599457eca86c0092

Hash del manifiesto Cheng (local_file, patient_id, tumor_type, subset, sha256), 16 chars:
  a78633a95254d7c2

SHA-256 completo del fichero manifest_final.csv usado como fuente:
  c468eb4e057dfab9b883b73c59e959f6194b0c46037df6b7cef22c9b2e5e6487
  (5872 filas totales, 3064 filas Cheng)

Referencia — huellas del dataset COMPLETO (binario) según dataset_audit.json
del run 70436 real, para detectar si en el futuro se usa otra versión del dataset:
  manifest_hash = b6881df313b8c6b5
  split_hash    = 81d2b363fa9c274a
  config_hash   = b67c4e3262df5b24
  (467 pacientes totales, 5872 imágenes, generado 2026-08-27 15:01:32)
```

Cualquier futura celda que cargue este split debe recalcular `SPLIT_HASH_MULTICLASE`
sobre el `manifest_final.csv` vigente y **detenerse** si no coincide con
`fda7e2daeb9ec2de`, en vez de regenerar un reparto nuevo.

## 7. Mapping exacto de clases

```python
TIPOS = sorted(cheng["tumor_type"].unique())  # orden alfabético
# TIPOS = ["glioma", "meningioma", "pituitario"]
MAPA_TIPO = {"glioma": 0, "meningioma": 1, "pituitario": 2}
```

## 8. Configuración exacta del baseline (modelo que dio 79,7 %)

- **Arquitectura:** `construir_cnn_scratch(n_salidas=3, activacion="softmax", nombre="cnn_3clases")` — la CNN propia del notebook (4 bloques conv 32/64/128/128 + conv final 256 + GAP + Dropout(0.4) + Dense(128, relu) + Dropout(0.3) + Dense(3, softmax)), **no** uno de los backbones preentrenados.
- **Compilación:** `Adam(learning_rate=1e-4)`, `loss="sparse_categorical_crossentropy"`, `metrics=["accuracy"]`.
- **Entrada:** 1 canal (escala de grises), tamaño `IMG_SIZE=(224,224)` (redimensionado bilineal desde el PNG de `PNG_SIZE=(256,256)`), normalización `x/255.0` — **sin** `preprocess_input` de ImageNet ni replicación a 3 canales (eso solo aplica a los backbones preentrenados, que Apéndice A no usa).
- **Data augmentation:** solo en train — `RandomFlip("horizontal")`, `RandomRotation(0.03)`, `RandomZoom(0.10)`, `RandomTranslation(0.05,0.05)`, `RandomContrast(0.10)`, todas sembradas con `SEED`.
- **Entrenamiento:** `EPOCHS_CNN=40` máximo, `EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)`. No hay `ReduceLROnPlateau` en esta celda concreta (sí se usa en `entrenar_o_cargar()`, pero Apéndice A no pasa por esa función).
- **Batch size:** 32 (`BATCH_SIZE`).
- **class_weight:** no se aplica en Apéndice A (a diferencia de la tarea binaria).
- **Selección de modelo:** ninguna — se entrena una sola vez, no hay comparación de arquitecturas ni CV para esta tarea de control en el run actual.

## 9. Método exacto de agregación slice → paciente

```python
pac3 = ap_te[["patient_id", "tumor_type"]].copy()
for i, t in enumerate(TIPOS):
    pac3[f"p_{t}"] = P[:, i]                     # prob. de cada slice, por clase
agr = pac3.groupby("patient_id").agg({**{f"p_{t}": "mean" for t in TIPOS},
                                      "tumor_type": "first"})
pred_pac = agr[[f"p_{t}" for t in TIPOS]].values.argmax(1)   # argmax de la media
real_pac = agr["tumor_type"].map(MAPA_TIPO).values
bal_pac = balanced_accuracy_score(real_pac, pred_pac)         # sklearn, multiclase
```

Esto coincide exactamente con la fórmula que pediste generalizar:
`P_paciente(c) = media de P_corte(c)` → `pred_paciente = argmax_c P_paciente(c)`.

**Nota importante sobre la métrica:** `balanced_accuracy_score` de scikit-learn en
modo multiclase es la **media del recall por clase**, no la media de
sensibilidad/especificidad como en la fórmula binaria del cuerpo del notebook.
Es la fórmula correcta y estándar para multiclase, pero hay que documentar que
"balanced accuracy" significa una cosa distinta en cada uno de los dos
experimentos (importante para la Sección 9 del plan — riesgos de comparar ambas
tareas sin aclarar esto).

## 10. Semilla(s)

- `SEED = 42` (config global, celda 1.1).
- `keras.utils.set_random_seed(SEED)` se vuelve a invocar justo antes de construir
  `modelo3`, además de la llamada ya hecha al principio de la Sección 2.
- `DETERMINISMO_ESTRICTO = False` en esta ejecución — es decir, **no** se activó
  `tf.config.experimental.enable_op_determinism()`. Con GPU y determinismo no
  estricto, TensorFlow puede introducir variación entre ejecuciones incluso con
  la misma seed (operaciones no deterministas en cuDNN). Esto significa que un
  reentrenamiento futuro con exactamente el mismo split/config puede no
  reproducir el 79,7 % en el último decimal, aunque sí debería quedar muy cerca
  si el pipeline es equivalente.

## 11. Checkpoint / predicciones del modelo del 79,7 % — HALLAZGO IMPORTANTE

**No existe ningún checkpoint guardado ni fichero de predicciones por paciente
para este modelo concreto.** Verificado explícitamente:

- `SHA256SUMS_70436.txtcd` solo lista `models/modelo_final.keras` — que es el
  modelo **binario** (MobileNetV2), no el de 3 clases.
- `predicciones_paciente_test.csv` (72 filas) contiene las predicciones del
  test **binario** (71 pacientes, columna `y` con 0/1) — no las de la tarea de
  control multiclase.
- `control_3clases.json` solo guarda 4 números resumen (clases, nº pacientes
  test, balanced accuracy, atajo de bajo nivel) — ninguna probabilidad por
  paciente, ninguna predicción individual, ningún artefacto `.keras`.

**Consecuencia práctica para la FASE 2B (regresión del baseline):** no es
posible verificar la equivalencia cargando un checkpoint o comparando
predicciones guardadas, como pedías. La única forma de comprobar que el
framework refactorizado reproduce el 79,7 % es **reentrenar** `cnn_scratch`
sobre este split congelado con esta configuración exacta, y comparar la
balanced accuracy resultante contra 79,7 % (con margen, dado el punto 10 sobre
determinismo no estricto). Esto sería un reentrenamiento de verificación de
código, no una decisión metodológica nueva, pero técnicamente sí implica
"reentrenar un modelo", así que lo señalo explícitamente antes de hacerlo.

## 12. Otras inconsistencias / puntos a tener en cuenta

1. **Alcance real del baseline "79,7 %":** este resultado existe **únicamente
   para `cnn_scratch`**. Apéndice A nunca ha entrenado VGG16, ResNet50,
   MobileNetV2 ni EfficientNetB0 sobre la tarea de 3 clases — esos cuatro
   backbones solo se han evaluado hasta ahora en la tarea binaria. Tu
   instrucción de FASE 2B dice "ejecutar/reutilizar exclusivamente los modelos
   que ya existían... y comprobar que la nueva implementación reproduce los
   resultados anteriores" para los cinco modelos — pero **solo hay resultado
   anterior que reproducir para `cnn_scratch`**. Para los otros cuatro, correrlos
   en `TASK="tumor_type"` sería un experimento nuevo, no una comprobación de
   regresión. Antes de ejecutar FASE 2B necesito que confirmes si quieres que
   trate esos cuatro como "primera vez" (documentado como tal, no como
   regresión) o si prefieres otra cosa.
2. El notebook `_1` que se te devolvió corregido y el que generó el 79,7 % no
   son el mismo fichero (ver punto 1), pero la comparación celda a celda
   confirma que son equivalentes en todo lo relevante para el split y para
   Apéndice A.
3. No se ha encontrado ninguna discrepancia entre `manifest_final.csv`, el
   Apéndice A y `control_3clases.json` en cuanto al split en sí: los 35
   pacientes de test cuadran exactamente, paciente a paciente.
