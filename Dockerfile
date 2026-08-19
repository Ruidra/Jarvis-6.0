# Jarvis — god-level image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps for audio/OCR/PyAutoGUI where applicable
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg portaudio19-dev tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-god.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-god.txt

COPY . .

# The neon/glass Web UI + phone UI (dashboard) on port 8000, including the
# /metrics health endpoint. The Gradio cross-device UI (web/app.py) is also
# available and can be run instead via:
#   python web/app.py
EXPOSE 8000
CMD ["python", "web/run_dashboard.py"]
