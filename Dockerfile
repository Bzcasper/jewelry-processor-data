# Multi-stage build for optimized production image
FROM python:3.9-slim
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=all

    
# Install Node.js and build frontend
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
    RUN apt-get install -y nodejs
    COPY frontend /app/frontend
    WORKDIR /app/frontend
    RUN npm install
    RUN npm run build
    
    # Copy built assets
COPY frontend/src/web/static/dist /app/src/web/static/dist

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create logging directory
RUN mkdir -p /var/log/jewelry-processor

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up logging configuration
COPY logging.conf /etc/jewelry-processor/logging.conf

# Expose ports
EXPOSE 5000

# Run the application with proper logging
CMD ["python3", "-u", "app.py"]