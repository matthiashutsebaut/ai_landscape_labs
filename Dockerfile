# Jupyter + MLflow for o3-system-taxonomy labs
#
# Build (from repo root):
#   docker build -t o3-labs -f excercises/o3-system-taxonomy/Dockerfile excercises/o3-system-taxonomy
#
# Run:
#   docker run -p 8888:8888 -p 5000:5000 o3-labs
#   - Jupyter: http://localhost:8888  (open lab-01, lab-02, or lab-03)
#   - MLflow:  http://localhost:5000
#
# Persist MLflow data and optionally mount labs for live edit:
#   docker run -p 8888:8888 -p 5000:5000 -v $(pwd)/mlflow-data:/workspace/mlflow o3-labs
#
FROM python:3.12-slim

WORKDIR /workspace

# System deps (optional; uncomment if you need compilers for some pip packages)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Lab dependencies + Jupyter + MLflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy labs and shared data
COPY data/                    /workspace/data/
COPY lab-01-the-model-zoo/    /workspace/lab-01-the-model-zoo/
COPY lab-02-enter-the-llm/    /workspace/lab-02-enter-the-llm/
COPY lab-03-how-models-learn/ /workspace/lab-03-how-models-learn/

# Ensure dataset exists (script writes to data/loan_applications.csv)
RUN python /workspace/data/generate_dataset.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Jupyter (8888), MLflow UI (5000)
EXPOSE 8888 5000

# Start MLflow server in background, then Jupyter
ENTRYPOINT ["/entrypoint.sh"]
