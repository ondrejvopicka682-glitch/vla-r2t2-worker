# RunPod serverless worker: netease-youdao/Confucius4-R2T2 (CUDA).
# Model weights are NOT baked into the image -- they're downloaded once into
# the attached Network Volume on first cold start (see handler.py), so the
# image itself stays small and rebuilds are fast.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3.11", "-u", "handler.py"]
