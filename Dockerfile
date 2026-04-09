FROM python:3.11-slim

# Install curl for healthcheck before pip install (better layer caching)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source into the package directory
COPY . /app/incident_response_env/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Enable the built-in Gradio web interface at /web (openenv-core feature)
ENV ENABLE_WEB_INTERFACE=true

# OpenEnv convention: container listens on 8000 (LocalDockerProvider hardcodes this).
# HF Spaces routes external traffic to this port via app_port: 8000 in README.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "incident_response_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
