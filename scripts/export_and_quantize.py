#!/usr/bin/env python3
"""
Model Export & Quantization Pipeline.
Downloads DistilBERT SST-2, exports to ONNX, applies Graph Optimization,
and dynamically quantizes weights to INT8.
"""

import os
import shutil
import logging
import onnx
from onnx import shape_inference
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTOptimizer
from optimum.exporters.onnx import main_export

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
    QUANT_ONNX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step 1: Exporting {MODEL_ID} to ONNX format...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    main_export(
        model_name_or_path=MODEL_ID,
        output=RAW_ONNX_DIR,
        task="text-classification",
        opset=17,
    )

    # Load the exported model without saving it back into RAW_ONNX_DIR.
    model = ORTModelForSequenceClassification.from_pretrained(
        RAW_ONNX_DIR,
        file_name="model.onnx",
    )

    tokenizer.save_pretrained(QUANT_ONNX_DIR)
    logger.info(f"Exported raw ONNX model to {RAW_ONNX_DIR}")

    optimizer = ORTOptimizer.from_pretrained(model)

    # O2 applies transformer attention fusion that is incompatible with the
    # ORT quantizer version used in this image. O1 retains safe graph
    # optimizations without producing untyped intermediate MatMul outputs.
    opt_config = AutoOptimizationConfig.O1()

    optimizer.optimize(
        save_dir=OPT_ONNX_DIR,
        optimization_config=opt_config,
    )

    optimized_model_path = OPT_ONNX_DIR / "model_optimized.onnx"

    optimized_files = list(OPT_ONNX_DIR.glob("*.onnx"))
    if len(optimized_files) != 1:
        raise RuntimeError(f"Expected one optimized ONNX file, found: {optimized_files}")

    optimized_model_path = optimized_files[0]

    # Restore inferred tensor types before quantization.
    inferred_model = shape_inference.infer_shapes(
        onnx.load(str(optimized_model_path))
    )
    onnx.save(inferred_model, str(optimized_model_path))

    quantizer.quantize(
        save_dir=QUANT_ONNX_DIR,
        quantization_config=dq_config,
    )

    # ORTQuantizer derives the output filename from the input model.
    # For example: model_optimized_quantized.onnx.
    quantized_files = list(QUANT_ONNX_DIR.glob("*_quantized.onnx"))

    if len(quantized_files) != 1:
        raise RuntimeError(
            f"Expected one quantized ONNX file, found: {quantized_files}"
        )

    generated_quantized_path = quantized_files[0]
    quantized_model_path = QUANT_ONNX_DIR / "model_quantized.onnx"

    if generated_quantized_path != quantized_model_path:
        generated_quantized_path.replace(quantized_model_path)

    # Compare sizes
    raw_size = os.path.getsize(
        RAW_ONNX_DIR / "model.onnx"
    ) / (1024 * 1024)

    quant_size = os.path.getsize(
        quantized_model_path
    ) / (1024 * 1024)

    logger.info("--- Export & Quantization Complete ---")
    logger.info(f"Raw ONNX Size:       {raw_size:.2f} MB")
    logger.info(f"Quantized INT8 Size: {quant_size:.2f} MB")
    logger.info(
        f"Reduction:           "
        f"{((raw_size - quant_size) / raw_size) * 100:.1f}%"
    )
