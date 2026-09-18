# ==========================================
# Stage 1: Builder & Quantizer
# ==========================================
FROM python:3.10-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install standard dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      --retries 10 \
      --timeout 180 \
      -r requirements.txt

# Copy the export script
COPY scripts/ scripts/



# Execute the pipeline: Download -> Optimize -> Quantize INT8
# This runs at build time, meaning the model is baked into the image.
RUN python scripts/export_and_quantize.py

# ==========================================
# Stage 2: Production Runtime
# ==========================================
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install \
      --retries 10 \
      --timeout 180 \
      --no-cache-dir \
      -r requirements.txt  
# Copy ONLY the fully optimized, quantized INT8 model from the builder
COPY --from=builder /app/model_store/onnx_quantized ./model_store/onnx_quantized

# Copy the application source code
COPY src/ ./src/

RUN addgroup --system mlops_user \
    && adduser --system --ingroup mlops_user mlops_user \
    && chown -R mlops_user:mlops_user /app
    
# Change ownership to non-root user
RUN chown -R mlops_user:mlops_user /app
USER mlops_user

# Expose the API port
EXPOSE 8000

# Start FastAPI using Uvicorn.
# IMPORTANT: workers=1 is intentional. The OnnxInferenceEngine manages its own 
# internal concurrency pool. Multiple Uvicorn workers would needlessly multiply memory overhead.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
