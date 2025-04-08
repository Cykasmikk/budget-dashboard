# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (optimized with cache mounts)
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app code (will be overwritten by Docker Compose volume during development)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app with file watcher enabled (auto reload)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.runOnSave=true", "--server.fileWatcherType=poll"]
