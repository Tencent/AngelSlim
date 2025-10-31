# Diffusion模型量化

AngelSlim 支持高效灵活的 Diffusion Transformer (DiT) FP8 量化，用户可自行配置量化方式和范围。

## 支持的FP8量化类型

- **fp8-per-tensor**：推荐，速度/精度平衡佳
- **fp8-per-tensor-weight-only**：仅权重量化，适合高精度场景
- **fp8-per-token**：粒度相较per-tensor更细，精度稍好
- **fp8-per-block**：适配DeepGEMM，粒度最细，精度最好

## 使用说明

- 通过 `DynamicDiTQuantizer` 配置量化类型及层筛选，灵活选取量化方式和范围。
- 量化模型即可直接用于推理，或导出量化权重与缩放因子便于部署。

详细用法请参考示例代码。
