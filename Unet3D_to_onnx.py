import torch
from Unet3D import UNet3D  # 导入你定义的UNet3D模型

def convert_to_onnx():
    # 初始化模型
    model = UNet3D(residual='pool')
    model.eval()  # 设置模型为评估模式

     # 定义新的输入尺寸
    new_batch_size = 2
    new_channels = 32
    new_depth = 64
    new_height = 64
    new_width = 64

    # 使用 torch.randn 生成随机输入张量
    input_tensor = torch.randn(new_batch_size, new_channels, new_depth, new_height, new_width)

    # 导出模型为ONNX格式
    onnx_path = "unet3d_model.onnx"
    torch.onnx.export(
        model,  # 要转换的模型
        input_tensor,  # 模型的输入示例
        onnx_path,  # 导出的ONNX文件路径
        export_params=True,  # 是否导出模型参数
        opset_version=17,  # ONNX算子集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入张量的名称
        output_names=['output'],  # 输出张量的名称
        dynamic_axes={
            'input': {0: 'batch_size'},  # 动态输入维度，这里指定批次维度为动态
            'output': {0: 'batch_size'}  # 动态输出维度，这里指定批次维度为动态
        }
    )
    print(f"模型已成功转换为ONNX格式，保存路径为: {onnx_path}")

if __name__ == '__main__':
    convert_to_onnx()