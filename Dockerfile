FROM python:3.11-slim

# Tesseract is only needed for scanned (image-only) PDFs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    CHROMA_DIR=/var/data/chroma_db \
    TABLES_DIR=/var/data/tables

# Render injects $PORT (default 10000). Must bind 0.0.0.0.
CMD streamlit run main.py \
    --server.port=${PORT:-10000} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.maxUploadSize=200
