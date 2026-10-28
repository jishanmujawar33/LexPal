FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    poppler-utils \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Create supervisord config to run both services
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor

COPY supervisord.conf /etc/supervisor/conf.d/lexpal.conf

# HuggingFace Spaces uses port 7860
EXPOSE 7860 8000

# Create data and db dirs
RUN mkdir -p /app/data /app/legal_db

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/lexpal.conf"]
