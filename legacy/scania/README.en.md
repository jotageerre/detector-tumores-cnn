**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# legacy/scania — historical interface prototype

Artefacts from the first interface prototype (*ScanIA*), dated 11/07/2026, before the dataset was rebuilt. They are kept **only as a historical reference** (Appendix F of the technical appendices).

| File | Status |
|---|---|
| `brain_viewer.html` | Interface template with placeholders for CSS, JS and volume data |
| `brain_viewer_1.css` | Complete stylesheet |
| `Dockerfile` | Container intended for Cloud Run (`gunicorn … main:app`) |

There is no backend associated with this Dockerfile and the original JavaScript logic file was empty. The prototype was **never** connected to the final multiclass model. The application that does work is [`desktop-client/`](../../desktop-client) + [`cloud-run-backend/`](../../cloud-run-backend), whose `main.py` already includes its own embedded 3D viewer.
