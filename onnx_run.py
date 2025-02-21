import onnxruntime as ort
import numpy as np

# 加载量化后的 ONNX 模型
model_path = "unet3d_model_XINT8_quant.onnx"
session = ort.InferenceSession(model_path)

# 获取模型的输入名称和形状
input_info = session.get_inputs()[0]
input_name = input_info.name
input_shape = input_info.shape

# 确定输入数据的形状
# 注意：如果形状中有动态维度（如 'batch_size'），需要手动指定一个合适的值
batch_size = 1
if isinstance(input_shape[0], str):  # 假设第一个维度是动态的批次维度
    input_shape = [batch_size] + input_shape[1:]

# 生成合适的随机输入张量
# 先以 uint8 类型生成随机数据
input_data_uint8 = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
# 将数据类型转换为 float32
input_data = input_data_uint8.astype(np.float32)

# 运行推理
outputs = session.run(None, {input_name: input_data})

# 输出结果
print("推理结果：", outputs)