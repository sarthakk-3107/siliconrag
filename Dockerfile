FROM python:3.11-slim

WORKDIR /app

# System deps for pdfplumber (needs libs for image extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY data/chroma_db/ ./data/chroma_db/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
