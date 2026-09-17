# High-Throughput DistilBERT Inference Server (ONNX + INT8)

This repository contains a production-grade MLOps deployment stack for DistilBERT. It prioritizes low-latency, high-throughput NLP inference (Sentiment Analysis) on CPU-only infrastructure using ONNX Runtime, Dynamic INT8 Quantization, and FastAPI.

##  Architecture Overview

The system is designed to handle concurrent incoming requests efficiently without hardware locking or Out-Of-Memory (OOM) crashes.

1. **Model Pipeline**: `HuggingFace DistilBERT` ➔ `ONNX Export` ➔ `Graph Optimization` ➔ `Dynamic INT8 Quantization` (~50% size reduction, ~2x speedup on CPU).
2. **Inference Engine**: A lock-free, thread-safe session pool prevents CPU thrashing during concurrent requests.
3. **Serving Layer**: FastAPI with Pydantic V2 validation to drop oversized/malicious payloads before they reach the inference engine.
4. **Observability**: Prometheus metrics track millisecond-level latency histograms and throughput.

##  Getting Started (Local Development)

### Prerequisites
* Python 3.10+
* Docker

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Export and Quantize the Model
Run the pipeline script to download the model from HuggingFace, apply ONNX graph optimizations, and perform INT8 quantization. This saves the artifacts locally to `model_store/`.

```bash
python scripts/export_and_quantize.py
```

### 3. Run the Server
The FastAPI server will automatically load the quantized model from `model_store/onnx_quantized`, run a warmup compilation phase, and bind to port 8000.

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

## Docker Deployment
The `Dockerfile` uses a multi-stage build. The model is downloaded, optimized, and quantized during the build phase. The final image only contains the lightweight runtime environment and the final `.onnx` artifacts, keeping the image small and secure.

```Bash
# Build the image (this will take a few minutes as it quantizes the model)
docker build -t distilbert-inference:latest .

# Run the container mapping port 8000
docker run -d -p 8000:8000 --name distilbert-api distilbert-inference:latest
```

### API Reference
Once the server is running (either locally or via Docker), the Swagger UI is available at `http://localhost:8000/docs`.

Predict (Single)
`POST /predict`

```JSON
{
  "text": "The integration between ONNX and FastAPI is incredibly smooth."
}
```
Predict (Batch)
`POST /predict/batch`
(Max 32 items per batch to prevent OOM)


```JSON
{
  "texts": [
    "I love this architecture.",
    "The cold start latency was terrible before we added the WarmupManager."
  ]
}
```
### Observability Endpoints
`GET /health`: Kubernetes liveness probe (checks if the model engine is loaded).

`GET /model/info`: Returns metadata about the deployed model architecture and quantization strategy.

`GET /metrics`: Exposes Prometheus-compatible metrics (model_inference_latency_seconds, model_inference_throughput_total, model_active_requests).

### CI/CD Pipeline
This project uses GitHub Actions (`.github/workflows/ci_cd.yml`) to enforce code quality and automate deployments.

Continuous Integration (CI): On every PR, the pipeline runs `ruff` (linting), `mypy` (type-checking), and `pytest`.

Continuous Deployment (CD): On merge to main, the pipeline builds the multi-stage `Docker` image, tags it, pushes it to the GitHub Container Registry (GHCR), and triggers a remote deployment script on your self-hosted server via SSH.

Required GitHub Secrets for CD:

`SERVER_HOST`: The IP address of your deployment server.

`SERVER_USER`: The SSH username.

`SERVER_SSH_KEY`: The private SSH key for authentication.
