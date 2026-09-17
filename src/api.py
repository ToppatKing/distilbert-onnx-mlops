import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .engine import OnnxInferenceEngine
from .schemas import (
    PredictRequest, 
    PredictionResult, 
    BatchPredictRequest, 
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse
)
from .metrics import ACTIVE_REQUESTS, MODEL_INFO

# Configure logging for the API layer
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global variable to hold our engine instance
engine: OnnxInferenceEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan manager.
    Handles startup (loading the model into memory) and shutdown logic.
    """
    global engine
    model_path = os.getenv("MODEL_PATH", "model_store/onnx_quantized")
    pool_size = int(os.getenv("POOL_SIZE", "4"))
    
    logger.info("Starting up FastAPI server...")
    try:
        # Initialize the ONNX Engine (this triggers compilation and warmup)
        engine = OnnxInferenceEngine(model_dir=model_path, pool_size=pool_size)
        
        # Register static model metadata for Prometheus
        MODEL_INFO.labels(
            model_id="distilbert-sst-2",
            quantization="Dynamic INT8",
            framework="ONNX Runtime"
        ).set(1)
        
        logger.info("Server startup complete. Ready to serve traffic.")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Inference Engine: {e}")
        raise RuntimeError("Failed to load model during startup.") from e
    finally:
        logger.info("Shutting down server, cleaning up resources...")
        # Since we use queue.Queue and standard C++ ORT sessions, Python's GC 
        # handles cleanup, but explicit cleanup logic could go here if needed.

# Initialize FastAPI App
app = FastAPI(
    title="DistilBERT ONNX Inference Server",
    description="High-throughput, INT8-quantized sentiment analysis API.",
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# Inference Endpoints
# ==========================================

@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictRequest):
    """Single inference endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine is not ready.")
    
    ACTIVE_REQUESTS.inc()
    try:
        # We run the synchronous ONNX inference directly. 
        # For a massive I/O bound service, we might use run_in_threadpool, 
        # but for fast CPU-bound ML inference, async/await overhead often outweighs benefits.
        result = engine.predict_single(request.text)
        return result
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error.")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Batched inference endpoint (up to 32 items)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine is not ready.")
    
    ACTIVE_REQUESTS.inc()
    try:
        results = engine.predict_batch(request.texts)
        return BatchPredictResponse(
            results=results,
            batch_size=len(results)
        )
    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal batch inference error.")
    finally:
        ACTIVE_REQUESTS.dec()


# ==========================================
# Observability & Management Endpoints
# ==========================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe for Kubernetes / Docker Swarm."""
    is_ready = engine is not None
    return HealthResponse(
        status="healthy" if is_ready else "initializing",
        model_loaded=is_ready
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return PlainTextResponse(
        generate_latest(), 
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Returns metadata about the currently loaded model."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine is not ready.")
    
    return ModelInfoResponse(
        model_name="HuggingFace DistilBERT SST-2",
        architecture="Transformer",
        quantization="Dynamic INT8 (ONNX)",
        pool_size=engine.pool_size,
        framework="ONNX Runtime CPU"
    )
