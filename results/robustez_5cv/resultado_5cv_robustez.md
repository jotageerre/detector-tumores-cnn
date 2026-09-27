# Resultado de validación interna de robustez 5-fold

## Contexto

Esta validación se realizó después de haber seleccionado previamente ResNet50 con fine-tuning mediante el benchmark OOF original.

No se utilizaron los resultados de esta 5-fold cross-validation para:

- seleccionar arquitectura;
- modificar hiperparámetros;
- modificar el número de épocas;
- modificar el número de capas descongeladas;
- buscar nuevas estrategias de fusión.

Por tanto, se trata de una validación interna complementaria de robustez del modelo ya seleccionado.

El resultado del protocolo original sobre el TEST congelado de 35 pacientes se conserva sin cambios:

- Balanced accuracy: 93,89 %
- Accuracy: 94,29 %
- Macro-F1: 93,98 %

## Cohorte de la validación 5CV

Cheng et al.:

- 233 pacientes
- 3064 cortes
- glioma: 89 pacientes
- meningioma: 82 pacientes
- pituitario: 62 pacientes

Se combinaron los pacientes pertenecientes originalmente a train, validación y TEST para generar un nuevo particionado independiente.

## Particionado

StratifiedGroupKFold:

- n_splits = 5
- groups = patient_id
- shuffle = True
- random_state = 20260904

Cada paciente aparece exactamente una vez como conjunto retenido de evaluación a lo largo de los cinco folds.

No existe solapamiento de patient_id entre entrenamiento y evaluación dentro de ningún fold.

## Modelo

Configuración previamente fijada:

ResNet50 preentrenado en ImageNet.

Fase 1:
- backbone congelado
- cabeza clasificadora nueva
- Adam lr=1e-4
- 6 épocas fijas

Fase 2:
- últimas 20 capas no-BatchNormalization descongeladas
- BatchNormalization congelado
- Adam lr=1e-5
- 6 épocas fijas

Semillas de entrenamiento:

- fold 1: 1001
- fold 2: 1002
- fold 3: 1003
- fold 4: 1004
- fold 5: 1005

## Resultados por fold

Fold 1:
- BA: 92,42 %
- Accuracy: 89,36 %
- Macro-F1: 90,11 %

Fold 2:
- BA: 82,30 %
- Accuracy: 80,85 %
- Macro-F1: 80,32 %

Fold 3:
- BA: 94,34 %
- Accuracy: 93,48 %
- Macro-F1: 94,29 %

Fold 4:
- BA: 90,07 %
- Accuracy: 89,36 %
- Macro-F1: 89,43 %

Fold 5:
- BA: 88,07 %
- Accuracy: 86,96 %
- Macro-F1: 87,07 %

## Media entre folds

Balanced accuracy:
89,44 % ± 4,64 puntos porcentuales

Accuracy:
88,00 % ± 4,63 puntos porcentuales

Macro-F1:
88,24 % ± 5,14 puntos porcentuales

La desviación típica es la desviación típica muestral entre los cinco folds.

## Predicciones OOF agregadas

Cada uno de los 233 pacientes fue evaluado exactamente una vez por un modelo que no había sido entrenado con ese paciente.

Balanced accuracy:
88,68 %

Accuracy:
87,98 %

Macro-F1:
88,21 %

Pacientes correctamente clasificados:
205 / 233

Recall por clase:

- glioma: 88,76 %
- meningioma: 80,49 %
- pituitario: 96,77 %

Matriz de confusión agregada, filas=clase real y columnas=predicción:

[[79, 9, 1],
 [9, 66, 7],
 [1, 1, 60]]

## Interpretación

La nueva validación muestra una variabilidad relevante entre particiones: la balanced accuracy varía desde 82,30 % hasta 94,34 %.

Esto confirma que el rendimiento observado en una única partición pequeña puede depender de los pacientes que hayan quedado retenidos.

El resultado histórico de 93,89 % BA sobre los 35 pacientes del TEST original continúa siendo válido para aquel holdout concreto, pero la 5CV proporciona una estimación complementaria más robusta de la variabilidad interna del modelo.

Meningioma es la clase con menor recall agregado (80,49 %), frente a 88,76 % en glioma y 96,77 % en pituitario.

## Limitación

Esta 5CV no constituye validación externa.

Incluye los 233 pacientes de Cheng, incluidos los 35 pacientes pertenecientes al TEST histórico. Por tanto debe describirse como validación interna post-selección del modelo y no como un nuevo test externo independiente.
