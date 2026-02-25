#!/usr/bin/env bash
set -e

# MLflow server (background): backend + artifact store under /workspace/mlflow
MLFLOW_DIR="${MLFLOW_DIR:-/workspace/mlflow}"
mkdir -p "$MLFLOW_DIR"
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
  --default-artifact-root "${MLFLOW_DIR}/mlruns" \
  &

# So notebooks (e.g. lab-03) log to this server
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

# Jupyter: listen on all interfaces so host can access via -p 8888:8888
echo "Jupyter will be at http://localhost:8888 (use the token from the log below)"
echo "MLflow UI will be at http://localhost:5000"
exec jupyter notebook \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.allow_remote_access=True \
  --ServerApp.allow_origin='*' \
  --NotebookApp.token="${JUPYTER_TOKEN:-}" \
  --NotebookApp.password="${JUPYTER_PASSWORD:-}" \
  --notebook-dir=/workspace
