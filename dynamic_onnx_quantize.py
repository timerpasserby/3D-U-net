import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 定义原始 ONNX 模型路径和量化后 ONNX 模型保存路径
original_onnx_model_path = "unet3d_model.onnx"
quantized_onnx_model_path = "unet3d_model_quantized.onnx"

# 进行动态量化
quantize_dynamic(
    model_input=original_onnx_model_path,
    model_output=quantized_onnx_model_path,
    weight_type=QuantType.QInt8
)

print(f"模型已成功量化，量化后的模型保存路径为: {quantized_onnx_model_path}")