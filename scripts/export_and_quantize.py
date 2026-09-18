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

    # Restore inferred tensor types before quantization.
    inferred_model = shape_inference.infer_shapes(
        onnx.load(str(optimized_model_path))
    )
    onnx.save(inferred_model, str(optimized_model_path))

    quantizer = ORTQuantizer.from_pretrained(OPT_ONNX_DIR)

    dq_config = AutoQuantizationConfig.avx512_vnni(
        is_static=False,
        per_channel=True,
    )

    quantizer.quantize(
        save_dir=QUANT_ONNX_DIR,
        quantization_config=dq_config,
    )


    # 4. Compare sizes
    raw_size = os.path.getsize(RAW_ONNX_DIR / "model.onnx") / (1024 * 1024)
    quant_size = os.path.getsize(QUANT_ONNX_DIR / "model_quantized.onnx") / (1024 * 1024)
    
    logger.info("--- Export & Quantization Complete ---")
    logger.info(f"Raw ONNX Size:       {raw_size:.2f} MB")
    logger.info(f"Quantized INT8 Size: {quant_size:.2f} MB")
    logger.info(f"Reduction:           {((raw_size - quant_size) / raw_size) * 100:.1f}%")

if __name__ == "__main__":
    main()
