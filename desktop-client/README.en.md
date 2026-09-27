**Language:** [Español](README.md) · **English** · [Français](README.fr.md)

# desktop-client — desktop application

Tkinter application that:

1. logs in with a local user;
2. lets you choose the model (**Cheng multiclass** or historical binary);
3. uploads an image (JPG/PNG) or a NIfTI volume to Google Cloud Storage;
4. calls `POST /predict` on the Cloud Run service;
5. shows the class, the probabilities and the Grad-CAM map, or opens the 3D viewer in the browser.

<p align="center"><img src="../docs/img/app_resultado.jpg" width="60%"></p>

## Requirements

- Python 3.10+ (Tkinter ships with the official Windows/macOS installer; on Linux: `sudo apt install python3-tk`).
- A deployed Cloud Run service and a service account allowed to upload to the buckets ([`docs/en/06_google_cloud_deployment.md`](../docs/en/06_google_cloud_deployment.md)).

## Installation

```bash
python -m venv myenv
myenv\Scripts\activate            # Windows  (Linux/macOS: source myenv/bin/activate)
pip install -r requirements.txt
```

## Configuration (none of this is uploaded to GitHub)

| What | How |
|---|---|
| Service account key | Copy the JSON into this folder as **`gcp-service-account.json`**, or set `GOOGLE_APPLICATION_CREDENTIALS` |
| API URL | `SCANIA_API_URL` variable, or *Settings* inside the app, or `config.json` (copy of `config.example.json`) |
| Buckets | `SCANIA_BUCKET_2D` and `SCANIA_BUCKET_3D` variables |

`credentials.json` (app users) and `config.json` are created automatically when the app is used. Passwords are stored in plain text: this is a demo, do not use real passwords.

## Run

```bash
python main.py
```

> Experimental tool. The result is not a medical diagnosis.
