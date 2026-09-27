**Idioma:** **Español** · [English](en/04_results.md) · [Français](fr/04_resultats.md)

# 4. Resultados

Todas las cifras proceden del Anexo D (ficheros de resultados verificados). Todas son **a nivel de paciente**.

## 4.1 Baseline histórico (CNN propia desde cero)

| Experimento | Balanced acc. | Conjunto |
|---|---|---|
| CNN propia (histórico) | 79,66 % | test histórico (35 pacientes) |
| Reproducción con el framework reconstruido | 74,44 % | mismo test (control de regresión) |

La diferencia (−5,21 pp) superó la tolerancia de ±0,03 y **el control de regresión detuvo automáticamente** una de las ejecuciones (job 70886). Se documenta como limitación.

## 4.2 Comparación de arquitecturas (backbone congelado)

3 folds agrupados por paciente sobre 198 pacientes · test bloqueado.

| Modelo | Balanced acc. OOF | Accuracy | Macro-F1 | Recall glioma | Recall mening. | Recall pitui. | Parámetros |
|---|---|---|---|---|---|---|---|
| **ResNet50** | **83,13 %** | 81,31 % | 82,06 % | 79,22 % | 72,22 % | 97,96 % | 23,9 M |
| InceptionV3 | 82,06 % | 80,30 % | 80,40 % | 85,71 % | 62,50 % | 97,96 % | 22,1 M |
| MobileNetV2 | 81,53 % | 79,29 % | 79,45 % | 77,92 % | 66,67 % | 100,00 % | 2,4 M |
| VGG16 | 78,94 % | 76,77 % | 75,98 % | 89,61 % | 47,22 % | 100,00 % | 14,8 M |
| EfficientNetB0 | 78,72 % | 76,26 % | 75,90 % | 79,22 % | 56,94 % | 100,00 % | 4,2 M |

## 4.3 Ajuste fino y fusión

| Modelo | Balanced acc. OOF | Accuracy | Macro-F1 |
|---|---|---|---|
| **ResNet50 con fine-tuning** | **88,54 %** | 87,37 % | 88,02 % |
| Fusión 50/50 (ResNet50‑FT + InceptionV3‑FT) | 88,29 % | 87,37 % | 87,92 % |
| InceptionV3 con fine-tuning | 85,76 % | 84,34 % | 84,85 % |

La fusión no mejora al mejor modelo individual → se selecciona **ResNet50‑FT**.

<p align="center"><img src="img/seleccion_arquitectura.jpg" width="60%"></p>

## 4.4 Evaluación final sobre el test reservado (una sola vez)

| Métrica | Valor |
|---|---|
| Pacientes / cortes | 35 / 474 |
| **Balanced accuracy** | **93,89 %** |
| Accuracy | 94,29 % |
| Macro-F1 | 93,98 % |
| Recall glioma / meningioma / pituitario | 91,67 % / 90,00 % / 100,00 % |
| Precisión glioma / meningioma / pituitario | 100,00 % / 90,00 % / 92,86 % |
| IC95 % bootstrap (2.000 réplicas) de la balanced acc. | [84,26 %, 100,00 %] |

Matriz de confusión (filas = real, columnas = predicho; glioma / meningioma / pituitario):

```
[[11, 1, 0],
 [ 0, 9, 1],
 [ 0, 0, 13]]
```

33 de 35 pacientes correctos. Errores: un meningioma predicho como pituitario (confianza 0,82, 8 cortes) y un glioma predicho como meningioma (confianza 0,46, la más baja del test, 2 cortes).

## 4.5 Validación cruzada de robustez (5 folds, 233 pacientes)

Hecha **después** de fijar el modelo, sin volver a tocar la selección. Un modelo nuevo por fold desde los pesos de ImageNet. Ficheros en [`results/robustez_5cv/`](../results/robustez_5cv).

| Fold | Semilla | Pac. train / eval | Balanced acc. | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| 1 | 1001 | 186 / 47 | 92,42 % | 89,36 % | 90,11 % |
| 2 | 1002 | 186 / 47 | 82,30 % | 80,85 % | 80,32 % |
| 3 | 1003 | 187 / 46 | 94,34 % | 93,48 % | 94,29 % |
| 4 | 1004 | 186 / 47 | 90,07 % | 89,36 % | 89,43 % |
| 5 | 1005 | 187 / 46 | 88,07 % | 86,96 % | 87,07 % |
| **Media ± DE** | | | **89,44 % ± 4,64 pp** | 88,00 % ± 4,63 pp | 88,24 % ± 5,14 pp |

Sobre las **233 predicciones agregadas** (cada paciente evaluado una vez por un modelo que no lo vio): balanced accuracy **88,68 %**, accuracy 87,98 %, macro-F1 88,21 %, 205/233 pacientes correctos.

```
[[79,  9,  1],
 [ 9, 66,  7],
 [ 1,  1, 60]]
```

<p align="center"><img src="img/folds_5cv.jpg" width="45%"> <img src="img/holdout_vs_5cv.jpg" width="45%"></p>

**Cómo leerlo:** el 93,89 % del test está cerca del extremo superior del rango de los folds (82,30 %–94,34 %). La estimación más prudente del rendimiento dentro de esta cohorte es la de la 5CV. La clase más difícil es **meningioma** (confusión con pituitario, sobre todo en el fold 2).

## 4.6 Grad-CAM

<p align="center"><img src="img/gradcam.jpg" width="55%"></p>

Cinco casos del test: tres aciertos (uno por clase) y los dos errores. En los aciertos la confianza es > 0,90; en uno de los errores el modelo se equivoca con confianza 0,82, lo que recuerda que **confianza alta no significa acierto**. Galería completa en el Anexo E.
