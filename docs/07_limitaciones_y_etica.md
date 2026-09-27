# 7. Limitaciones y consideraciones éticas

Fuente: capítulos 11 y 13 de la memoria.

## Qué NO demuestra este trabajo

- **No hay validación externa.** Todo se evaluó dentro de la cohorte de Cheng (233 pacientes, dos hospitales chinos, T1 con contraste). No se sabe cómo se comporta el modelo con otros escáneres, protocolos, hospitales o poblaciones.
- **No hay validación clínica.** No es un dispositivo médico ni una herramienta de diagnóstico.
- **El test es pequeño.** 35 pacientes: un solo paciente mal clasificado mueve la balanced accuracy varios puntos. De ahí el intervalo de confianza tan ancho [84,26 %, 100 %] y la importancia de la validación de 5 folds.
- **La 5CV no es anidada.** La arquitectura se eligió antes con otro particionado; la 5CV mide la variabilidad del modelo ya elegido, no el proceso de selección completo.
- **Grad-CAM es cualitativo.** No se comparó con las máscaras tumorales, así que no prueba que el modelo mire el tumor.
- **Posibles atajos dentro de Cheng.** Un clasificador de bajo nivel obtiene un 69,22 % en la tarea multiclase (el azar sería 33 %). No se ha demostrado que el modelo dependa de ello, pero tampoco se ha descartado.
- **Sin determinismo estricto de GPU.** Las reejecuciones no darán las mismas cifras exactas.
- **Parte del código de la selección de arquitectura no se conserva** (la versión exacta del notebook con el fine-tuning, la fusión y la evaluación final; Anexo A.3).

## Datos

| Aspecto | Cheng | IXI |
|---|---|---|
| Anonimización | Declarada explícitamente por los autores | No verificada con el mismo detalle |
| Aprobación ética | Declarada (comités del Nanfang Hospital y del General Hospital de la Tianjin Medical University) | No verificada con el mismo detalle |
| Consentimiento informado | No se localizó mención explícita | No verificado |
| Licencia | CC BY 4.0 (verificada con reserva) | CC BY-SA 3.0, con obligación de citar |

Este repositorio **no redistribuye imágenes**: solo identificadores de paciente del propio dataset público (p. ej. `CHENG-100360`), que son necesarios para reproducir el split.

## La app de escritorio

- Las contraseñas de usuarios locales se guardan **en texto plano** en `credentials.json` (limitación real del demostrador, no una característica de seguridad).
- Los resultados (Grad-CAM, visor 3D) se publican en buckets de lectura pública.
- No hay registro persistente de qué imagen, modelo y resultado se generó en cada inferencia.

Por todo ello, **no uses la app con datos de pacientes reales**.

## Trabajo futuro propuesto

1. Validación externa en una cohorte independiente (otro centro, otro escáner).
2. Validación cruzada anidada (selección de modelo dentro de la estimación).
3. Evaluación cuantitativa de Grad-CAM frente a las máscaras tumorales de Cheng.
4. Buscar señales residuales de centro/escáner dentro de la cohorte y probar otras reglas de agregación por paciente.
5. Determinismo de GPU, entorno en contenedor reproducible y empaquetado completo.
6. En la app: mismo preprocesado que en el entrenamiento, trazabilidad de cada inferencia y almacenamiento privado.
