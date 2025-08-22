# Benchmark Instructions

本指南介绍如何使用 **PyTorch** 运行基线模型和 **Eagle3** 模型的推理基准测试，以及如何计算 **接受长度** 与 **加速比**。

---

## 目录
- [1. 运行 Baseline 模型基准测试（PyTorch）](#1-运行-baseline-模型基准测试pytorch)
- [2. 运行 Eagle3 模型基准测试（PyTorch）](#2-运行-eagle3-模型基准测试pytorch)
- [3. 计算接受长度与加速比](#3-计算接受长度与加速比)
  - [3.1 计算接受长度](#31-计算接受长度)
  - [3.2 计算加速比](#32-计算加速比)

---

## 1. 运行 Baseline 模型基准测试（PyTorch）

```bash
cd angelslim/compressor/speculative_decoding/scripts
sh run_baseline_benchmark_pytorch.sh
```

## 2. 运行 Eagle3 模型基准测试（PyTorch）
```bash
cd angelslim/compressor/speculative_decoding/scripts
sh run_eagle3_benchmark_pytorch.sh
```

## 3. 计算接受长度与加速比
### 3.1 计算接受长度
```bash
python3 get_alpha_and_speed.py acceptance --input_file [results.jsonl]
```
参数说明
*  `--input_file`：Eagle3 模型推理benchmark结果文件

### 3.2 计算加速比
```bash
python3 get_alpha_and_speed.py speedup --model_path /path/to/model --baseline_json [baseline.jsonl] --eagle_json [eagle.jsonl]
```
参数说明
* `--model_path`：Baseline 模型路径
* `--baseline_json`：Baseline 模型的benchmark结果文件
* `--eagle_json`：Eagle3 模型的benchmark结果文件
