# Image for the Kafka producer (and, later, the dashboard).
# Build from the repo root:
#   docker build -f docker/app.Dockerfile -t realtime-sentiment-app .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY models/ models/
COPY data/raw/ data/raw/
