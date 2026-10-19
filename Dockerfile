FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-сборка torch: образ в несколько раз меньше, чем с CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY moderation ./moderation
COPY configs ./configs

# модели монтируются в /app/models (make docker-run) — образ не зависит от весов
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s CMD python -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "moderation.api:app", "--host", "0.0.0.0", "--port", "8000"]
