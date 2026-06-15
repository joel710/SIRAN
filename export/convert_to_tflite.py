import argparse
import os

import numpy as np
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from core.model import build_siran_model


def export_to_onnx(checkpoint: str, output_path: str):
    model = build_siran_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported: {output_path}")


def quantize_onnx(input_path: str, output_path: str):
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Quantized model: {output_path} ({size_mb:.2f} MB)")


def export_to_tflite(checkpoint: str, output_path: str):
    import tensorflow as tf

    onnx_tmp = output_path.replace(".tflite", "_tmp.onnx")
    export_to_onnx(checkpoint, onnx_tmp)

    converter = tf.lite.TFLiteConverter.from_saved_model(onnx_tmp)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model exported: {output_path} ({size_mb:.2f} MB)")
    os.remove(onnx_tmp)


def main():
    parser = argparse.ArgumentParser(description="SIRAN Model Export & Quantization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", choices=["onnx", "tflite"], default="onnx")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--output", type=str, default="./export/siran_v1")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.format == "onnx":
        onnx_path = f"{args.output}.onnx"
        export_to_onnx(args.checkpoint, onnx_path)
        if args.quantize:
            quantize_onnx(onnx_path, f"{args.output}_int8.onnx")
    else:
        export_to_tflite(args.checkpoint, f"{args.output}.tflite")


if __name__ == "__main__":
    main()
