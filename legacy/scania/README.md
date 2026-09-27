**Idioma:** **Español** · [English](README.en.md) · [Français](README.fr.md)

# legacy/scania — prototipo histórico de interfaz

Artefactos del primer prototipo de interfaz (*ScanIA*), fechados el 11/07/2026, antes de la reconstrucción del dataset. Se conservan **solo como referencia histórica** (Anexo F de los anexos técnicos).

| Fichero | Estado |
|---|---|
| `brain_viewer.html` | Plantilla de interfaz con marcadores de posición para CSS, JS y datos del volumen |
| `brain_viewer_1.css` | Hoja de estilos completa |
| `Dockerfile` | Contenedor pensado para Cloud Run (`gunicorn … main:app`) |

No hay backend asociado a este Dockerfile y el fichero de lógica JavaScript original estaba vacío. El prototipo **no** se conectó nunca al modelo multiclase final. La aplicación que sí funciona es [`desktop-client/`](../../desktop-client) + [`cloud-run-backend/`](../../cloud-run-backend), cuyo `main.py` ya incluye su propio visor 3D incrustado.
