# training/

Código para preparar los datos, entrenar y evaluar. La guía completa, con salidas esperadas, está en [`docs/05_guia_reproduccion.md`](../docs/05_guia_reproduccion.md).

| Ruta | Qué es | Origen |
|---|---|---|
| `requirements.txt` | Entorno (TensorFlow 2.15.1, scikit-learn < 1.8) | añadido |
| `herramientas/construir_manifest_cheng.py` | Descarga Cheng de figshare, genera los PNG con el preprocesado original y el `manifest_final.csv` con el split congelado (verifica la huella `fda7e2da…`) | añadido |
| `entrenar_modelo_final.py` | Reimplementación del protocolo C.1: entrena ResNet50‑FT con 198 pacientes y evalúa una vez los 35 de test; exporta `.keras` y `.h5` | añadido |
| `notebooks/02_pipeline_patient_level_multiclase.ipynb` | Pipeline completo: descarga Cheng + IXI, manifiesto, duplicados, split congelado, baseline, benchmark de 5 arquitecturas, tarea binaria histórica | **original** |
| `notebooks/01_reconstruccion_dataset_y_control_3clases.ipynb` | Versión de la etapa de reconstrucción (job 70436), con salidas | **original** |
| `notebooks/baseline_multiclase_cheng.md` | Documento que congeló el split multiclase y el baseline del 79,7 % | **original** |
| `robustez_5cv/*.py`, `robustez_5cv/*.slurm` | Validación cruzada de 5 folds tal como se ejecutó (jobs 71101, 71102, 71107) | **original** |
| `slurm/run_TFG_…_multiclase.slurm` | Lanzador SLURM del notebook 02 | **original** |

Los scripts originales tienen escrita la ruta del clúster (`/shared/home/…/tfg_run`). Para usarlos, sustitúyela por la tuya (la guía indica el comando `sed`). No se han modificado aquí porque cada resultado guarda el SHA‑256 del script que lo generó.

Inicio rápido:

```bash
pip install -r requirements.txt
python herramientas/construir_manifest_cheng.py --root ./tfg_run --split ../data/splits/split_multiclase_cheng.csv
python entrenar_modelo_final.py --root ./tfg_run
```

Ambos scripts añadidos se han probado de principio a fin con datos sintéticos que imitan la estructura de Cheng (misma lista de pacientes y cortes): el manifiesto reproduce la huella del split y el entrenamiento, el guardado `.h5` y la evaluación funcionan. No se han podido ejecutar con los datos y pesos reales desde el entorno donde se preparó el repositorio.
