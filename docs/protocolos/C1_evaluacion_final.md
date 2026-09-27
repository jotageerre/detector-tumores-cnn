> Transcripción literal del protocolo fijado por escrito **antes de abrir el conjunto de test** (Anexo C.1 de los anexos técnicos).

# Protocolo de evaluación final — tarea multiclase (glioma/meningioma/pituitario)

## Modelo seleccionado
Resultado de la selección por balanced accuracy patient-level OOF (GroupCV 3-fold,
train+validación, 198 pacientes):
- ResNet50 fine-tuned: 0,8853930461
- Fusión 50/50 (ResNet50-FT + InceptionV3-FT): 0,8829193293
- InceptionV3 fine-tuned: 0,8576152683

Por tanto se selecciona ResNet50 fine-tuned. No se probarán otros pesos de fusión
ni configuraciones después de esta selección.

## Entrenamiento final
- Cohorte: 198 pacientes (train+val) para entrenamiento; 35 pacientes de test,
  bloqueados durante toda la selección de arquitectura.
- Split congelado (hash): fda7e2daeb9ec2de.

Fase 1 (cabeza):
- ResNet50 preentrenado en ImageNet, backbone congelado.
- Cabeza: GAP + Dropout(0.4) + Dense(128, ReLU) + Dropout(0.3) + Dense(3, softmax).
- Adam, lr = 1e-4.
- 6 épocas (número derivado como mediana de las mejores épocas de los folds CV: [5, 9, 6]).

Fase 2 (fine-tuning):
- Últimas 20 capas no-BatchNormalization descongeladas.
- BatchNormalization congelado.
- Adam, lr = 1e-5.
- 6 épocas (número derivado como mediana de las mejores épocas de fine-tuning: [7, 3, 6]).
- Batch size: 32.
- Entrada: 224×224, RGB mediante replicación del canal de escala de grises.
- Preprocesamiento: preprocess_input de ResNet50.
- Data augmentation: mismo esquema que en el benchmark de selección.
- Semilla: 42.

## Evaluación
- El conjunto de TEST se evalúa una única vez.
- Métrica principal: balanced accuracy a nivel de paciente.
- Métricas secundarias: accuracy, macro-F1, recall por clase, matriz de confusión.
- Agregación: media de probabilidades por corte a nivel de paciente, después argmax
  entre las 3 clases.

No se modificarán hiperparámetros después de observar TEST.
