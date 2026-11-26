# 其他部署体验方案

[← Strata 部署](./strata.md) ｜ [← 返回总览](./index.md)

除 llama.cpp 与 Strata 外，以下方案也可用于验证或部署 AngelSlim 的量化模型。

## 1. prima.cpp 多机部署（Hy4 preview）

Hy4 preview [MIX-STQ1_0 GGUF](https://huggingface.co/AngelSlim/Hy4-preview-GGUF)，可以在两台 PC 组成的集群上,利用prima.cpp运行。

流程概览：

1. 两台机器分别解压 prima.cpp 运行包，检查依赖；
2. 在每台机器上生成本机的 receipt 与 GPU 拓扑文件；
3. 生成共享执行计划，并分别启动 rank-1 与 rank-0；
4. rank-0 对外提供 OpenAI 兼容接口。

完整的命令、环境变量与参数见 [Hy4 Preview GGUF 部署指南](https://github.com/Tencent/AngelSlim/blob/main/docs/source/_extra/hy4_preview_gguf_guideline.md)。

## 2. 移动手机端体验。

Hy-MT1.5 提供了面向手机的离线应用：

- **Android Demo**：[Hy-MT-demo.apk](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF/resolve/main/Hy-MT-demo.apk)，支持完全离线翻译，并提供后台取词模式。
- **Arm SME2 设备**：Hy-MT1.5 的 2-bit 版本提供了 SME2 内核，并以 NEON 内核作为对照。编译与切换方式见 [Hy-MT1.5 文档](../../models/Hy-MT1.5/hy-mt1.5.md)。