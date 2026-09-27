# Validación de robustez 5-fold del modelo seleccionado

## Propósito

Esta fase NO realiza selección de arquitectura ni optimización de hiperparámetros.

El modelo ya seleccionado previamente mediante validación OOF sobre train+validación es:

ResNet50 con fine-tuning limitado.

La evaluación final original sobre el TEST congelado de 35 pacientes permanece
inalterada y conserva su resultado:

- balanced accuracy: 0.9388888889
- accuracy: 0.9428571429
- macro-F1: 0.9398282340

La presente fase proporciona una estimación interna complementaria de robustez
mediante 5-fold cross-validation sobre los 233 pacientes de Cheng.

## Cohorte

- Fuente: figshare/Cheng2017
- 233 pacientes
- 3064 slices
- glioma: 89 pacientes
- meningioma: 82 pacientes
- pituitario: 62 pacientes

Se combinan exclusivamente para esta validación los pacientes que pertenecían al
train, validación y test del split histórico congelado.

## Partición nueva

StratifiedGroupKFold:
- n_splits = 5
- groups = patient_id
- estratificación por tumor_type a nivel de paciente
- shuffle = True
- random_state = 20260904

El split histórico original NO se modifica.

## Modelo y entrenamiento

Arquitectura e hiperparámetros fijados previamente:

### Fase 1
- ResNet50 ImageNet
- backbone congelado
- cabeza: GAP + Dropout(0.4) + Dense(128, ReLU) + Dropout(0.3) + Dense(3, softmax)
- Adam, lr = 1e-4
- 6 épocas fijas

### Fase 2
- fine-tuning limitado
- últimas 20 capas no-BatchNormalization según el mismo procedimiento del modelo final
- BatchNormalization congelado
- Adam, lr = 1e-5
- 6 épocas fijas

No se utiliza EarlyStopping para decidir el número de épocas de cada fold.
El número de épocas queda fijado antes de ejecutar la 5CV.

## Semillas de entrenamiento

- fold 1: 1001
- fold 2: 1002
- fold 3: 1003
- fold 4: 1004
- fold 5: 1005

No se activa determinismo GPU estricto, manteniendo el comportamiento del
protocolo experimental anterior.

## Evaluación

Cada paciente aparece exactamente una vez en un fold retenido de evaluación.

Por fold:
- balanced accuracy patient-level
- accuracy patient-level
- macro-F1 patient-level
- recall por clase
- matriz de confusión
- predicciones por paciente

Resumen:
- media y desviación típica entre folds
- matriz de confusión acumulada
- métricas globales sobre las 233 predicciones OOF de pacientes

Esta 5CV es una validación interna de robustez del modelo ya seleccionado y no
una validación externa independiente.
