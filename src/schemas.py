from pydantic import BaseModel, Field
from typing import List

# ==========================================
# Requests
# ==========================================

class PredictRequest(BaseModel):
    """Schema for a single inference request."""
    # Constrain input length to prevent tokenization/OOM explosions
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="The text sequence to classify."
    )

class BatchPredictRequest(BaseModel):
    """Schema for batched inference requests."""
    texts: List[str] = Field(
        ..., 
        min_length=1,
        max_length=32, # Max 32 items per batch as defined in the architecture
        description="List of text sequences to classify."
    )
    
    # Add a validator to ensure individual strings aren't too long
    # (Pydantic V2 syntax for nested constraints can also be used, but keeping it explicit here is good practice)
    
# ==========================================
# Responses
# ==========================================

class PredictionResult(BaseModel):
    """Standardized output for a single prediction."""
    label: str = Field(..., description="The predicted class label (e.g., POSITIVE).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of the prediction.")

class BatchPredictResponse(BaseModel):
    """Output for a batch request."""
    results: List[PredictionResult]
    batch_size: int
    
class HealthResponse(BaseModel):
    """Schema for Kubernetes/Docker liveness probes."""
    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    """Schema for the /model/info metadata endpoint."""
    model_name: str
    architecture: str
    quantization: str
    pool_size: int
    framework: str
