# 端侧部署教程（总览）

本教程介绍如何把模型打包为 GGUF 并部署到端侧设备上。获取 GGUF 模型有两种方式：一是自行用 llama.cpp 把 HuggingFace BF16 模型转换并量化为 GGUF；二是直接下载 AngelSlim 量化好的 GGUF.

## 章节

| 章节 | 内容 |
| --- | --- |
| [打包 GGUF + 量化](./gguf_quant.md) | 用 llama.cpp 把 Hugging Face 模型转换为 GGUF 并量化 |
| [llama.cpp 部署](./llamacpp.md) | 编译 llama.cpp 并部署 GGUF 模型 |
| [Strata 部署](./strata.md) | 高性能 GGUF 推理，支持offload  |
| [其他部署体验方案](./other.md) | prima.cpp 多机部署、移动端体验 |

## 已发布的端侧 GGUF 模型

点击各系列展开模型列表。

<details>
<summary><strong>Hy4-flash</strong>：IQ2_AH、Q3_K、Q4_K_M、SPARK-4Bit 等</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy4-flash-IQ2_AH](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | IQ2_AH | 135 GB |
| [Hy4-flash-Q3_K](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | Q3_K | 155.4 GB |
| [Hy4-flash-Q4_K_M](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | Q4_K_M | 168.4 GB |
| [Hy4-flash-Q4_K_M_MTP_Q8_0](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | Q4_K_M + MTP (Q_8) | 164.4 GB |
| [Hy4-flash-Q4_K_M_MTP_Q8_0_Prune25](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) |剪枝 25% Q4_K_M + MTP (Q_8) | 141.5 GB |
| [Hy4-flash-SPARK-4Bit](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | SPARK-4Bit | 163 GB |
| [Hy4-flash-SPARK-4Bit-Fast](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | SPARK-4Bit（极速版） | 160 GB |
| [hy4-flash_dfly_Q8_0](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | Q8_0（DFly Draft） | 904 MB |
| [mmproj-Hy4-flash-bf16](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | BF16（视觉投影） | 1.04 GB |

</details>
<details>
<summary><strong>Hy4</strong>：Q4_K_M、IQ2、MIX-STQ1_0</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy4-IQ2](https://huggingface.co/AngelSlim/Hy4-GGUF) | IQ2 | 332 GB |
| [Hy4-Q4_K_M](https://huggingface.co/AngelSlim/Hy4-GGUF) | Q4_K_M | 470 GB |
| [Hy4-STQ1_0](https://huggingface.co/AngelSlim/Hy4-GGUF) | MIX-STQ1_0 | 230 GB |
| [mmproj-HY4-VL-BF16](https://huggingface.co/AngelSlim/Hy4-GGUF) | BF16（视觉投影） | 2.68 GB |

</details>
<details>
<summary><strong>Hy-MT3</strong>：1.8B、7B、30B-A3B</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy-MT3-1.8B-Q4_K_M](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) | Q4_K_M | 1.35 GB |
| [Hy-MT3-1.8B-Q8_0](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) | Q8_0 | 1.91 GB |
| [Hy-MT3-7B-IQ2](https://huggingface.co/tencent/Hy-MT3-7B-GGUF) | IQ2 | 3.87 GB |
| [Hy-MT3-7B-Q4_K_M](https://huggingface.co/tencent/Hy-MT3-7B-GGUF) | Q4_K_M | 5.15 GB |
| [Hy-MT3-7B-Q8_0](https://huggingface.co/tencent/Hy-MT3-7B-GGUF) | Q8_0 | 8.73 GB |
| [Hy-MT3-30B-A3B-IQ2](https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF) | IQ2 | 13.1 GB |
| [Hy-MT3-30B-A3B-Q4_K_M](https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF) | Q4_K_M | 19.5 GB |
| [Hy-MT3-30B-A3B-Q8_0](https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF) | Q8_0 | 32.8 GB |

</details>
<details>
<summary><strong>Hy4-preview</strong>：Q4_K_M、UD-IQ1_M、MIX-STQ1_0</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy4-preview-Q4_K_M](https://huggingface.co/AngelSlim/Hy4-preview-GGUF) | Q4_K_M | 467 GB |
| [Hy4-preview-UD-IQ1_M](https://huggingface.co/AngelSlim/Hy4-preview-GGUF) | UD-IQ1_M | 235 GB |
| [Hy4-preview-STQ1_0](https://huggingface.co/AngelSlim/Hy4-preview-GGUF) | MIX-STQ1_0 | 229 GB |

</details>
<details>
<summary><strong>HY-MT2</strong>：1.8B、7B、30B-A3B（标准量化），1.8B（1.25bit、2bit）</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy-MT2-1.8B-Q4_K_M](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | Q4_K_M | 1.13 GB |
| [Hy-MT2-1.8B-Q6_K](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | Q6_K | 1.47 GB |
| [Hy-MT2-1.8B-Q8_0](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | Q8_0 | 1.91 GB |
| [Hy-MT2-7B-Q4_K_M](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) | Q4_K_M | 4.62 GB |
| [HY-MT2-7B-Q6_K](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) | Q6_K | 6.16 GB |
| [HY-MT2-7B-Q8_0](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) | Q8_0 | 7.98 GB |
| [Hy-MT2-30B-A3B-Q4_K_M](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) | Q4_K_M | 18.2 GB |
| [Hy-MT2-30B-A3B-Q8_0](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF) | Q8_0 | 32 GB |
| [Hy-MT2-1.8B-1.25Bit](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) | STQ | 462 MB |
| [Hy-MT2-1.8B-2Bit](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF) | 2-bit | 601 MB |

</details>
<details>
<summary><strong>HY-MT1.5</strong>：1.8B（1.25bit、2bit）</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy-MT1.5-1.8B-1.25bit](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF) | STQ1_0 | 462 MB |
| [Hy-MT1.5-1.8B-2bit](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-2bit-GGUF) | 2-bit | 601 MB |

</details>
<details>
<summary><strong>Hy-SimulMT</strong>：30B-A3B（translation-only、interleave）</summary>

| 模型 | 量化格式 | 大小 |
| --- | --- | --- |
| [Hy-SimulMT-30B-A3B-Q4_K_M](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF) | Q4_K_M | 19.7 GB |
| [Hy-SimulMT-30B-A3B-Q8_0](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-GGUF) | Q8_0 | 33.3 GB |
| [Hy-SimulMT-30B-A3B-Interleave-Q4_K_M](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF) | Q4_K_M | 19.7 GB |
| [Hy-SimulMT-30B-A3B-Interleave-Q8_0](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B-Interleave-GGUF) | Q8_0 | 33.3 GB |

</details>

各模型的获取与部署方式见 [打包 GGUF + 量化](./gguf_quant.md) 和后续章节。
