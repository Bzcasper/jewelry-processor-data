# Build stage for frontend
FROM node:16-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend .
RUN npm run build

# Production stage
FROM python:3.9-slim

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create logging directory
RUN mkdir -p /var/log/jewelry-processor \
    && mkdir -p /etc/jewelry-processor

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy frontend build from previous stage
COPY --from=frontend-build /app/frontend/build /app/build

# Copy application code and logging configuration
COPY . .
COPY logging.conf /etc/jewelry-processor/logging.conf

# Expose port
EXPOSE 5000

# Run the application with proper logging
CMD ["python3", "-u", "app.py"]