# ============================================================
# ScanIA backend — Cloud Run
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias de sistema mínimas (Pillow / matplotlib / nibabel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Código + assets del visor (templates/ y static/ deben copiarse)
COPY . .

# Cloud Run inyecta PORT (por defecto 8080)
ENV PORT=8080

# 1 worker (el modelo se cachea en memoria por instancia); varios hilos para I/O.
# timeout alto por los NIfTI grandes.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
