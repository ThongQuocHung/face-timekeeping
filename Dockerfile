FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (FIXED - replaced libgl1-mesa-glx with libgl1)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1

# Expose port (Railway will provide PORT env var)
EXPOSE 8080

# Start command with gunicorn
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --max-requests 100 \
    --max-requests-jitter 20 \
    --worker-class sync \
    --worker-tmp-dir /dev/shm \
    --log-level info