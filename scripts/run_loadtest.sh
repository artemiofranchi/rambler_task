#!/usr/bin/env bash
# Поднимает сервис в фоне, ждёт готовности, гоняет нагрузку (одиночные запросы и батчи), останавливает сервис.
set -euo pipefail
PY=${PY:-python3}
PORT=${PORT:-8000}
DURATION=${DURATION:-30}
CONCURRENCY=${CONCURRENCY:-32}
URL="http://127.0.0.1:${PORT}"

$PY -m uvicorn moderation.api:app --host 127.0.0.1 --port "$PORT" --log-level warning &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "$URL/health" >/dev/null || { echo "сервис не поднялся"; exit 1; }

$PY scripts/loadtest.py --url "$URL" --mode single --concurrency "$CONCURRENCY" --duration "$DURATION"
$PY scripts/loadtest.py --url "$URL" --mode batch --batch-size 64 --concurrency 4 --duration "$DURATION"
