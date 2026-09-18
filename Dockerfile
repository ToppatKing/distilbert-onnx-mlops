# ==========================================
# Stage 1: Builder & Quantizer
# ==========================================
FROM python:3.10-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install standard dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the export script
COPY scripts/ scripts/

# Execute the pipeline: Download -> Optimize -> Quantize INT8
# This runs at build time, meaning the model is baked into the image.
RUN python scripts/export_and_quantize.py

# ==========================================
# Stage 2: Production Runtime
# ==========================================
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="/app/model_store/onnx_quantized"
ENV POOL_SIZE="4"

WORKDIR /app

# Create a non-root user for security
RUN useradd -m -s /bin/bash mlops_user

# Install only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ONLY the fully optimized, quantized INT8 model from the builder
COPY --from=builder /app/model_store/onnx_quantized ./model_store/onnx_quantized

# Copy the application source code
COPY src/ ./src/

# Change ownership to non-root user
RUN chown -R mlops_user:mlops_user /app
USER mlops_user

# Expose the API port
EXPOSE 8000

# Start FastAPI using Uvicorn.
# IMPORTANT: workers=1 is intentional. The OnnxInferenceEngine manages its own 
# internal concurrency pool. Multiple Uvicorn workers would needlessly multiply memory overhead.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
