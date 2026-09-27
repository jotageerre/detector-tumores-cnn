**Idioma:** **Español** · [English](README.en.md) · [Français](README.fr.md)

# results/

Resultados originales de la **validación de robustez de 5 folds** (jobs 71102 y 71107), copiados sin modificar. No se incluyen los modelos (`.keras`, ~214 MB cada uno).

| Fichero | Contenido |
|---|---|
| `robustez_5cv/protocolo_5cv_robustez.md` | Protocolo escrito antes de ejecutar |
| `robustez_5cv/resultado_5cv_robustez.md` | Resumen redactado tras la agregación |
| `robustez_5cv/resultado_5cv_validacion_robustez.json` | Métricas por fold, media ± DE y métricas globales sobre las 233 predicciones |
| `robustez_5cv/resultado_5cv_fold_{1..5}.json` | Resultado completo de cada fold (historial de entrenamiento, capas descongeladas, hashes) |
| `robustez_5cv/resumen_5cv_folds.csv` | Una fila por fold |
| `robustez_5cv/confusion_matrix_5cv_agregada.csv` | Matriz de confusión de los 233 pacientes |
| `robustez_5cv/predicciones_paciente_5cv_oof_233.csv` | Probabilidades y predicción de cada paciente, evaluado por un modelo que no lo vio |

Los resultados de la selección de arquitectura y de la evaluación final sobre el test están en el Anexo D ([`docs/04_resultados.md`](../docs/04_resultados.md)); sus ficheros originales forman parte de un paquete de 16 GB que no se sube al repositorio.
