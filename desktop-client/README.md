# desktop-client — aplicación de escritorio

Aplicación Tkinter que:

1. inicia sesión con un usuario local;
2. deja elegir modelo (**Cheng multiclase** o binario histórico);
3. sube una imagen (JPG/PNG) o un volumen NIfTI a Google Cloud Storage;
4. llama a `POST /predict` del servicio Cloud Run;
5. muestra la clase, las probabilidades y el mapa Grad-CAM, o abre el visor 3D en el navegador.

<p align="center"><img src="../docs/img/app_resultado.jpg" width="60%"></p>

## Requisitos

- Python 3.10+ (Tkinter viene con el instalador oficial de Windows/macOS; en Linux: `sudo apt install python3-tk`).
- Un servicio Cloud Run desplegado y una cuenta de servicio con permiso para subir a los buckets ([`docs/06_despliegue_google_cloud.md`](../docs/06_despliegue_google_cloud.md)).

## Instalación

```bash
python -m venv myenv
myenv\Scripts\activate            # Windows  (Linux/macOS: source myenv/bin/activate)
pip install -r requirements.txt
```

## Configuración (nada de esto se sube a GitHub)

| Qué | Cómo |
|---|---|
| Clave de la cuenta de servicio | Copia el JSON en esta carpeta como **`gcp-service-account.json`**, o define `GOOGLE_APPLICATION_CREDENTIALS` |
| URL de la API | Variable `SCANIA_API_URL`, o *Settings* dentro de la app, o `config.json` (copia de `config.example.json`) |
| Buckets | Variables `SCANIA_BUCKET_2D` y `SCANIA_BUCKET_3D` |

`credentials.json` (usuarios de la app) y `config.json` se crean solos al usarla. Las contraseñas se guardan en texto plano: es un demostrador, no uses contraseñas reales.

## Ejecutar

```bash
python main.py
```

> Herramienta experimental. El resultado no es un diagnóstico médico.
