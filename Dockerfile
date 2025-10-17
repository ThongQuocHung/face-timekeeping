# Sử dụng Python 3.11.10
FROM python:3.11.10-slim

# Cài đặt dependencies hệ thống
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Cài đặt Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2 app:app