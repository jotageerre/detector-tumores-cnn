> Transcripción literal del protocolo definido antes de evaluar la fusión (Anexo C.4).

# Fase 2C — Fusión multiclase

Protocolo definido antes de evaluar la fusión.

Modelos:
- ResNet50 con fine-tuning limitado
- InceptionV3 con fine-tuning limitado

Predicciones:
- exclusivamente OOF de train+val
- 3-fold StratifiedGroupKFold por patient_id
- test permanece bloqueado

Fusión única:
- soft voting 50/50
- p_fusion = 0.5 * p_ResNet50 + 0.5 * p_InceptionV3

Agregación:
- media de probabilidades por paciente
- argmax entre glioma, meningioma y pituitario

Métrica principal:
- balanced accuracy a nivel de paciente

Secundarias:
- macro-F1
- accuracy
- recall por clase

No se optimizarán los pesos de la fusión después de observar el resultado.
El ganador se seleccionará por balanced accuracy patient-level OOF.
