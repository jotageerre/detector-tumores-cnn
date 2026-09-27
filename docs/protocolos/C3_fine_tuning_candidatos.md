> Transcripción literal del protocolo fijado antes de ejecutar el ajuste fino (Anexo C.3).

# Fase 2B — Fine-tuning limitado multiclase

Protocolo fijado antes de ejecutar los experimentos.

## Selección de candidatos
Se seleccionan exclusivamente los dos mejores backbones según balanced accuracy
patient-level OOF del benchmark GroupCV:
1. ResNet50: 0.8313406171
2. InceptionV3: 0.8205782313

MobileNetV2 se conserva como alternativa eficiente, pero no se somete a fine-tuning.

## Fine-tuning
- Split congelado: fda7e2daeb9ec2de
- StratifiedGroupKFold: 3 folds
- Agrupación: patient_id
- TEST: bloqueado
- Inicialización: checkpoint de base congelada correspondiente al mismo fold
- Capas descongeladas: últimas 20 capas del backbone que no sean BatchNormalization
- BatchNormalization: permanece congelado
- Cabeza clasificadora: entrenable
- Learning rate: 1e-5
- Épocas máximas: 8
- EarlyStopping: val_accuracy, mode=max, patience=3, restore_best_weights=True
- ReduceLROnPlateau: val_loss, factor=0.3, patience=2, min_lr=1e-7
- Batch size: 32
- Data augmentation: mismo esquema ya definido
- Métrica principal de selección: balanced accuracy a nivel de paciente
- Métricas secundarias: macro-F1 y recall por clase
- Agregación paciente: media de probabilidades por clase + argmax

No se probarán otros learning rates, cantidades de capas o candidatos después de
observar estos resultados.
