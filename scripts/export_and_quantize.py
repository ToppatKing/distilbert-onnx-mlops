#!/usr/bin/env python3
"""
Model Export & Quantization Pipeline.
Downloads DistilBERT SST-2, exports to ONNX, applies Graph Optimization,
and dynamically quantizes weights to INT8.
"""

import os
import shutil
import logging
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
BASE_DIR = Path("model_store")
RAW_ONNX_DIR = BASE_DIR / "onnx_raw"
OPT_ONNX_DIR = BASE_DIR / "onnx_optimized"
QUANT_ONNX_DIR = BASE_DIR / "onnx_quantized"

def clean_directories():
    """Ensure clean slate for model export."""
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True)
    logger.info("Cleaned model directories.")

def main():
    clean_directories()

    # 1. Export HuggingFace Model to ONNX
    logger.info(f"Step 1: Exporting {MODEL_ID} to ONNX format...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_ID, export=True)
    
    # Save tokenizer to the final directory right away
    tokenizer.save_pretrained(QUANT_ONNX_DIR)
    model.save_pretrained(RAW_ONNX_DIR)
    logger.info(f"Exported raw ONNX model to {RAW_ONNX_DIR}")

    # 2. Graph Optimization
    logger.info("Step 2: Applying ONNX Graph Optimizations (fusion, constant folding)...")
    optimizer = ORTOptimizer.from_pretrained(model)
    # ORT optimization config for sequence classification
    opt_config = AutoOptimizationConfig.O2() # O2 includes basic + extended + layout optimizations
    optimizer.optimize(save_dir=OPT_ONNX_DIR, optimization_config=opt_config)
    logger.info(f"Saved optimized graph to {OPT_ONNX_DIR}")

    # 3. Dynamic INT8 Quantization
    # Note: Dynamic quantization is optimal for NLP models on CPUs
    logger.info("Step 3: Applying Dynamic INT8 Quantization...")
    # Load the optimized model for quantization
    quantizer = ORTQuantizer.from_pretrained(OPT_ONNX_DIR)
    
    # avx2/avx512 configs are standard for dynamic quant on modern x86 CPUs
    dq_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    
    quantizer.quantize(
        save_dir=QUANT_ONNX_DIR,
        quantization_config=dq_config
    )
    logger.info(f"Saved Dynamic INT8 quantized model to {QUANT_ONNX_DIR}")

    # 4. Compare sizes
    raw_size = os.path.getsize(RAW_ONNX_DIR / "model.onnx") / (1024 * 1024)
    quant_size = os.path.getsize(QUANT_ONNX_DIR / "model_quantized.onnx") / (1024 * 1024)
    
    logger.info("--- Export & Quantization Complete ---")
    logger.info(f"Raw ONNX Size:       {raw_size:.2f} MB")
    logger.info(f"Quantized INT8 Size: {quant_size:.2f} MB")
    logger.info(f"Reduction:           {((raw_size - quant_size) / raw_size) * 100:.1f}%")

if __name__ == "__main__":
    main()
