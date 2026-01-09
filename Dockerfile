FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies required for FAISS, Pillow, CLIP
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Azure Web App listens on 8000
EXPOSE 8000

# Start FastAPI
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "--bind", "0.0.0.0:8000"]
