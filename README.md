# Clasificación multiclase de tumores cerebrales en RM con Deep Learning

**Trabajo Fin de Grado** · Grado en Ingeniería Informática – Ingeniería del Software · Universidad de Sevilla (ETSII), 2026
Autor: **Joaquín González Rodríguez** · Tutor: José Cristóbal Riquelme Santos · Cotutor: Manuel Carranza García

Este repositorio reúne **todo lo necesario para entender, reproducir y reutilizar** el TFG: el pipeline de entrenamiento con trazabilidad por paciente, los splits congelados, los protocolos experimentales, los resultados, un servicio de inferencia en **Google Cloud Run** y una **aplicación de escritorio** que lo consume.

> ⚠️ **Uso exclusivamente experimental y educativo.** El modelo se ha evaluado dentro de una única cohorte pública (Cheng et al., 233 pacientes), **sin validación externa ni clínica**. No es un dispositivo médico ni sirve para diagnosticar a nadie.

---

## En 30 segundos

| | |
|---|---|
| **Tarea** | Clasificar un corte de RM T1 con contraste en **glioma**, **meningioma** o **tumor pituitario** |
| **Datos** | [Cheng et al. (figshare)](https://doi.org/10.6084/m9.figshare.1512427) · 233 pacientes · 3.064 cortes · CC BY 4.0 |
| **Modelo final** | **ResNet50** preentrenado en ImageNet + ajuste fino de las últimas 20 capas (ResNet50‑FT) |
| **Unidad de evaluación** | El **paciente** (media de probabilidades de sus cortes + argmax) |
| **Test reservado (35 pacientes, evaluado una vez)** | Balanced accuracy **93,89 %** · Accuracy 94,29 % · Macro‑F1 93,98 % · IC95 % bootstrap [84,26 %, 100 %] |
| **Validación cruzada 5 folds (233 pacientes)** | Balanced accuracy **89,44 % ± 4,64 pp** entre folds · 88,68 % sobre las 233 predicciones agregadas |
| **Demo** | [Vídeo (2 min, ES/EN)](https://youtu.be/QgTOD3KOTpI) |

<p align="center"><img src="docs/img/app_resultado.jpg" width="48%"> <img src="docs/img/gradcam.jpg" width="40%"></p>

## ¿Por qué este proyecto puede serte útil?

Más allá de la cifra de rendimiento, el valor del TFG está en **cómo** se llegó a ella. Si vas a entrenar una CNN con imágenes médicas, estos son los errores que el proyecto encontró (y corrigió) por el camino:

1. **Dividir por imagen en lugar de por paciente.** Los datasets agregados de Kaggle no traen identificador de paciente: cortes del mismo paciente pueden acabar en train y en test. Aquí se reconstruyó el dataset desde las fuentes primarias usando el `PID` real de cada fichero.
2. **Sesgo de procedencia.** La tarea binaria "tumor / no tumor" combinaba Cheng (tumor) e IXI (sano). Un clasificador con **solo 10 descriptores de bajo nivel, sin red neuronal**, obtuvo un **98,16 %** de accuracy: la etiqueta coincidía con el hospital de origen. Por eso el proyecto se reorientó a clasificación multiclase dentro de una sola cohorte.
3. **Abrir el test más de una vez.** Aquí el protocolo de evaluación se escribió y se fijó con su hash *antes* de abrir el test, y el test se evaluó una sola vez.

Todo esto se explica paso a paso en [`docs/01_historia_del_proyecto.md`](docs/01_historia_del_proyecto.md).

---

## Estructura del repositorio

```
detector-tumores-cnn/
├── docs/                      Documentación: memoria y anexos completos (PDF) + guías en Markdown
│   ├── 01_historia_del_proyecto.md   Evolución, errores detectados y decisiones
│   ├── 02_datos.md                   Fuentes, licencias, manifiesto, duplicados, split
│   ├── 03_metodologia.md             Preprocesado, arquitecturas, entrenamiento, métricas, Grad-CAM
│   ├── 04_resultados.md              Todas las tablas de resultados
│   ├── 05_guia_reproduccion.md       ★ Cómo recrear el entrenamiento paso a paso
│   ├── 06_despliegue_google_cloud.md ★ Cómo montar tu propio Cloud Run + app de escritorio
│   ├── 07_limitaciones_y_etica.md
│   └── protocolos/                   Protocolos C.1–C.4 tal como se fijaron antes de cada experimento
├── training/                  Código de entrenamiento y evaluación
│   ├── notebooks/             Notebooks originales del pipeline (Colab / HPC)
│   ├── robustez_5cv/          Scripts y SLURM originales de la validación cruzada de 5 folds
│   ├── slurm/                 Script SLURM del benchmark multiclase
│   ├── herramientas/          construir_manifest_cheng.py (añadido para el repo)
│   └── entrenar_modelo_final.py      Reimplementación del protocolo C.1 (añadido para el repo)
├── data/                      Split congelado por paciente + folds 5CV (sin imágenes)
├── results/                   Resultados de la validación 5CV (JSON/CSV)
├── cloud-run-backend/         API Flask de inferencia (servicio run-model)
├── desktop-client/            Aplicación de escritorio Tkinter
└── legacy/scania/             Prototipo histórico de interfaz (solo referencia)
```

## Empezar

**Solo quiero entender el trabajo** → lee [`docs/01_historia_del_proyecto.md`](docs/01_historia_del_proyecto.md) y [`docs/04_resultados.md`](docs/04_resultados.md), o la memoria completa en [`docs/TFG_Joaquin_Gonzalez_Rodriguez_memoria.pdf`](docs/TFG_Joaquin_Gonzalez_Rodriguez_memoria.pdf).

**Quiero reentrenar el modelo** → [`docs/05_guia_reproduccion.md`](docs/05_guia_reproduccion.md). Versión corta:

```bash
cd training
pip install -r requirements.txt            # Python 3.10, TensorFlow 2.15.1
python herramientas/construir_manifest_cheng.py --root ./tfg_run --split ../data/splits/split_multiclase_cheng.csv
python entrenar_modelo_final.py --root ./tfg_run   # unos minutos en una GPU tipo A30; horas en CPU
```

**Quiero desplegar la API y usar la app** → [`docs/06_despliegue_google_cloud.md`](docs/06_despliegue_google_cloud.md).

## Modelos entrenados

Los pesos no están en git (el modelo final ocupa ~170 MB y GitHub no admite ficheros de más de 100 MB). Se publican en la sección **Releases** del repositorio:

| Fichero | Modelo |
|---|---|
| `resnet50_ft_cheng_final.h5` | ResNet50‑FT multiclase que sirve el servicio `run-model` |
| `brain_tumor_cnn.h5` | CNN binaria histórica (afectada por el sesgo de procedencia; solo para la demo) |

## Qué es original y qué se ha añadido

Para que nadie confunda lo que produjo los resultados del TFG con lo que se ha preparado después para publicarlo:

| Tipo | Ficheros |
|---|---|
| **Artefactos originales del TFG** (sin modificar) | `training/notebooks/*.ipynb`, `training/robustez_5cv/*`, `training/slurm/*`, `data/splits/*`, `results/robustez_5cv/*`, `docs/protocolos/*`, `docs/*.pdf`, `legacy/scania/*`, `cloud-run-backend/main.py` |
| **Añadido para el repositorio** | `training/herramientas/construir_manifest_cheng.py`, `training/entrenar_modelo_final.py`, todos los `README.md` y `docs/*.md`, `desktop-client/main.py` (solo cambia la configuración: rutas relativas en vez de absolutas y URL/buckets configurables por variable de entorno), `cloud-run-backend/Procfile` |

El notebook exacto que ejecutó el ajuste fino y la evaluación final (jobs 70929–70931) **no se conserva** (Anexo A.3). `entrenar_modelo_final.py` reconstruye ese protocolo a partir del protocolo C.1 y del script real de la validación 5CV, que usa exactamente la misma receta.

## Cómo citar

```
González Rodríguez, J. (2026). Clasificación multiclase de tumores cerebrales en imágenes de
resonancia magnética mediante aprendizaje profundo [Trabajo Fin de Grado]. Universidad de Sevilla.
```

Si usas los datos, cita también la fuente original: Cheng, J. (2024). *Brain Tumor Dataset.* figshare. https://doi.org/10.6084/m9.figshare.1512427.v8 (CC BY 4.0), y el artículo que los autores piden citar en la página del dataset.
