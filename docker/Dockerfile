# =============================================================================
# Medical Prediction Dashboard - Docker Configuration
# Optimized for NVIDIA Jetson Orin (ARM64 architecture)
# =============================================================================

# Base image - Python 3.12 slim with security updates
FROM python:3.12-slim-bookworm

# Metadata
LABEL maintainer="Tom Huang"
LABEL description="Real-time Medical Prediction Dashboard for Hemodialysis Patients"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies and security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose dashboard port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Default command - run dashboard
CMD ["python", "-m", "src.dashboard.app"]
