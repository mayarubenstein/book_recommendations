#!/usr/bin/env bash
set -euo pipefail

python /app/backend/prepare_space_data.py

python -m uvicorn api:app --host 127.0.0.1 --port 8000 &
backend_pid=$!

cleanup() {
  kill "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd /app/frontend
exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true
