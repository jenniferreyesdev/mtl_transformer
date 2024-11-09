# Use bullseye slim as base image for better compatibility
FROM python:3.11.5-slim-bullseye AS builder

# Set working directory
WORKDIR /app

# Install build dependencies and pip tools in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install -U pip setuptools wheel

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -r requirements.txt

# Final stage
FROM python:3.11.5-slim-bullseye

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy only the necessary source code
COPY . .

# Default command
CMD ["python", "train.py"]