简体中文 | [English](README_en.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/source/assets/logos/angelslim_logo_light.png">
    <img alt="AngelSlim" src="./docs/source/assets/logos/angelslim_logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
致力于打造更易用、更全面和更高效的大模型压缩工具包
</h3>

<p align="center">
          📖 <a href="https://angelslim.readthedocs.io/">Documentation</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/AngelSlim">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/AngelSlim">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="./docs/source/assets/angel_slim_wechat.png">WeChat (微信)</a> | &nbsp&nbsp🫨 <a href="https://discord.com/invite/dHVNeuNdFt">Discord</a>
<br>
</p>


## 目录
- [最新进展](#最新进展)
- [主要特性](#主要特性)
- [支持模型](#支持模型)
- [如何使用](#如何使用)
  - [安装 AngelSlim](#安装-AngelSlim)
  - [快速开始](#快速开始)
  - [部署与测试](#部署与测试)
- [Benchmark](#benchmark)
- [许可协议](#许可协议)
- [引用](#引用)
- [技术交流](#技术交流)

## 📣最新进展
- 🌟[25/09/30] 我们开源了思考早退算法 SpecExit 的实现：*SpecExit: Accelerating Large Reasoning Model via Speculative Exit* | [[论文]](http://arxiv.org/abs/2509.24248) | [[文档]](https://angelslim.readthedocs.io/zh-cn/latest/features/speculative_decoding/spec_exit.html)
- 🌟[25/09/30] 我们发布了三值量化 Tequila 的实现：*TEQUILA: TRAPPING-FREE TERNARY QUANTIZATION FOR LARGE LANGUAGE MODELS* | [[论文]](https://arxiv.org/abs/2509.23809) | [[代码]](https://github.com/Tencent/AngelSlim/tree/tequila/TernaryQuant)。
- [25/09/24] 我们支持了Qwen3系列模型的NVFP4的PTQ量化，我们还开源了[Qwen3-32B-NVFP4](https://huggingface.co/AngelSlim/Qwen3-32B_nvfp4)、[Qwen3-235B-A22B-NVFP4](https://huggingface.co/AngelSlim/Qwen3-235B-A22B_nvfp4)权重。
- [25/09/01] 我们支持了[Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8)翻译开源模型的FP8量化；支持了Eagle3的Torch推理及Benchmark评测流程；支持了[FLUX](https://github.com/Tencent/AngelSlim/tree/main/configs/flux)的量化、Cache；支持了[Seed-OSS](https://github.com/Tencent/AngelSlim/tree/main/configs/seed_oss)模型量化压缩。
- [25/08/06] 我们支持了`Hunyuan 0.5B/1.8B/4B/7B`和`Qwen2.5VL 3B/7B/32B/72B`的FP8、INT4量化，支持了`DeepSeek-R1/V3`和`Kimi-K2`模型的`FP8-Static`、`W4A8-FP8`量化。我们还开源了`Hunyuan 1.8B/4B/7B`系列模型的Eagle3权重。
- [25/07/04] 我们支持了`Hunyuan/Qwen2.5/Qwen3/DeepSeek-R1-Distill-Qwen`等模型的量化，包含INT8、FP8、INT4等算法。
我们还开源了`Qwen3`系列模型的Eagle3权重。

Coming soon：
- [ ] Diffusion模型压缩支持
- [ ] 投机采样新算法发布

## 🌟主要特性

- **高度集成化**：本工具将主流的压缩算法集成到工具，开发者可一键式调用，具有很好的易用性。
- **持续算法创新**：本工具除了集成工业界使用最广的算法，还持续自研更好的压缩算法，并且会陆续开源。
- **追求极致性能**：在模型压缩流程、压缩算法部署方面，本工具持续端到端优化，例如单卡GPU可量化Qwen3-235B和Deepseek-R1。

## 💼支持模型

### 量化
目前已支持文生文任务Hunyuan-Dense、Hunyuan-MoE、Qwen3-Dense、Qwen3-MoE、Qwen2.5、DeepSeek-R1蒸馏Qwen模型、QwQ等系列的主要模型：

| 模型名      | FP8-Dynamic       | FP8-Static        | INT8-Dynamic | INT4-GPTQ         | INT4-AWQ          |
| ---------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| [Hunyuan-Dense](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)   |      ✅           |         ✅           | ✅           |    ✅               |         ✅           |
| [Hunyuan-MoE](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be)   |      ✅           |         ✅           | ✅           |    ✅               |         ✅           |
| [Qwen3-Dense](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)               |      ✅           |         ✅           | ✅           |    ✅               |         ✅           |
| [Qwen3-MoE](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)            |      ✅           |         ✅           | ✅           |     ✅             |        ✅            |
| [Qwen2.5](https://huggingface.co/collections/AngelSlim/qwen2-25-quant-68652d6cbdf5c0d4b1c4499a)            |      ✅           |         ✅           | ✅           |     ✅             |        ✅            |
| [DeepSeek-R1-Distill-Qwen](https://huggingface.co/collections/AngelSlim/deepseek-r1-distill-quant-68652f16a9c206b030b05f7f) |      ✅           |         ✅           | ✅           |      ✅             |        ✅            |
| [QwQ](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8) |      ✅           |         ✅           |       ✅             | ✅           |       ✅            |

### 投机采样

#### Eagle3
目前已开源Qwen3和Hunyuan系列模型的Eagle3权重。

| Qwen3  Models   | Hunyuan Models     |
| ----------|----------|
| ✅ [Qwen3-1.7B](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3)    |✅ [Hunyuan-1.8B-Instruct](https://huggingface.co/AngelSlim/Hunyuan-1.8B-Instruct_eagle3)    |
| ✅ [Qwen3-4B](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)        |✅ [Hunyuan-4B-Instruct](https://huggingface.co/AngelSlim/Hunyuan-4B-Instruct_eagle3)        |
| ✅ [Qwen3-8B](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)        |✅ [Hunyuan-7B-Instruct](https://huggingface.co/AngelSlim/Hunyuan-7B-Instruct_eagle3)        |
| ✅ [Qwen3-14B](https://huggingface.co/AngelSlim/Qwen3-14B_eagle3)      |
| ✅ [Qwen3-32B](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3)      |
| ✅ [Qwen3-30B-A3B](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3)  |





## 🛎️如何使用

### 安装 AngelSlim

推荐使用`pip`直接安装最新稳定版`AngelSlim`：

```shell
pip install angelslim
```

也可以选择克隆代码仓库后，以可编辑的方式从源代码安装：

```shell
cd AngelSlim && python setup.py install
```

更详细的安装说明可参考[安装文档](https://angelslim.readthedocs.io/zh-cn/latest/getting_started/installation.html)。

### 快速开始

#### 量化
完成安装`AngelSlim`后，您可以通过以下脚本快速开始，完成`Qwen3-1.7B`模型的静态`FP8`量化：

- 一键式启动

  ```shell
  python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-1_7b_fp8_static.yaml
  ```

  该示例将会加载`HugggingFace`模型， 使用`config`配置的`dataset`数据进行激活值校准，量化产出模型权重.

- 源码启动

  对`Qwen3-1.7B`完成动态`FP8`量化：

  ```python
  from angelslim.engine import Engine

  slim_engine = Engine()
  # Prepare model
  slim_engine.prepare_model(model_name="Qwen", model_path="Qwen/Qwen3-1.7B",)
  # Initialize compressor
  slim_engine.prepare_compressor("PTQ", default_method="fp8_dynamic")
  # Compress model
  slim_engine.run()
  # Save compressed model
  slim_engine.save("./output")
  ```

详情请参考[快速开始文档](https://angelslim.readthedocs.io/zh-cn/latest/getting_started/quickstrat.html)。

#### 投机采样
完成安装`AngelSlim`后，您可以通过以下脚本快速开始，完成`Eagle3`的Pytorch性能测试：
```bash
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id your_model_id \
    --mode both
```

详情请参考[快速开始文档](https://angelslim.readthedocs.io/zh-cn/latest/getting_started/quickstrat.html)。

### 部署与测试

#### 1. 离线推理

如果需要通过`transformers`加载量化模型，请在量化模型配置的`global`中设置`deploy_backend: huggingface`，或者直接手动将量化产出模型路径下`config.json`配置中的key `ignored_layers`改为`ignore`。

测试`transformers`加载量化模型离线推理：

```shell
python deploy/offline.py $MODEL_PATH
```

其中 `MODEL_PATH` 为量化产出模型路径。

#### 2. 服务部署

支持通过以下推理框架部署 OpenAI 兼容的 API 服务：

**vLLM**

[vLLM](https://github.com/vllm-project/vllm) 服务启动脚本，建议版本`vllm>=0.8.5.post1`，部署MOE INT8量化模型需要`vllm>=0.9.2`。

```shell
bash deploy/run_vllm.sh $MODEL_PATH
```

**SGLang**

[SGLang](https://github.com/sgl-project/sglang) 服务启动脚本，建议版本 `sglang>=0.4.6.post1`：

```shell
bash deploy/run_sglang.sh $MODEL_PATH
```

#### 3. 服务调用

通过 [OpenAI 格式](https://platform.openai.com/docs/api-reference/introduction) 接口发起请求：

```shell
bash deploy/openai.sh $MODEL_PATH
```

#### 4. 效果验证

使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 评估量化模型精度，建议版本`lm-eval>=0.4.8`：

```shell
bash deploy/lm_eval.sh $MODEL_PATH
```

详细操作指南请参阅[部署文档](https://angelslim.readthedocs.io/zh-cn/latest/deployment/deploy.html)。

## 📈Benchmark

### （1）量化

下面只展示了部分模型的效果测试情况，完整Benchmark可以参考[Benchmark文档](https://angelslim.readthedocs.io/zh-cn/latest/performance/quantization/benchmarks.html)

#### Hunyuan系列模型

Hunyuan-Instruct的`BF16`、`FP8`、`INT4-GPTQ`、`INT4-AWQ`在`OlympiadBench`、`AIME 2024`、`DROP`、`GPQA-Diamond`上的评测结果如下：

<table>
  <thead>
    <tr><th>Model</th><th>Quantization</th><th>OlympiadBench</th><th>AIME 2024</th><th>DROP</th><th>GPQA-Diamond</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="4">Hunyuan-A13B-Instruct</td>
    <td>BF16</td><td>82.7</td><td>87.30</td><td>91.1</td><td>71.2</td></tr>
    <tr><td>FP8-Static</td><td>83.0</td><td>86.7</td><td>91.1</td><td>-</td></tr>
    <tr><td>Int4-GPTQ</td><td>82.7</td><td>86.7</td><td>91.1</td><td>-</td></tr>
    <tr><td>Int4-AWQ</td><td>82.6</td><td>85.6</td><td>91.0</td><td>-</td></tr>
  </tbody>
  <tbody>
    <tr><td rowspan="4">Hunyuan-7B-Instruct</td>
    <td>BF16</td>          <td>76.5</td><td>81.1</td><td>85.9</td><td>60.1</td></tr>
    <tr><td>FP8-Static</td><td>76.6</td><td>80.9</td><td>86.0</td><td>60.1</td></tr>
    <tr><td>Int4-GPTQ</td><td>76.2</td><td>81.0</td><td>85.7</td><td>60.0</td></tr>
    <tr><td>Int4-AWQ</td><td>76.4</td><td>80.9</td><td>85.9</td><td>60.1</td></tr>
  </tbody>
  <tbody>
    <tr><td rowspan="4">Hunyuan-4B-Instruct</td>
    <td>BF16</td>          <td>73.1</td><td>78.3</td><td>78.2</td><td>61.1</td></tr>
    <tr><td>FP8-Static</td><td>73.1</td><td>76.6</td><td>78.3</td><td>60.2</td></tr>
    <tr><td>Int4-GPTQ</td><td>72.9</td><td>-</td><td>78.1</td><td>58.1</td></tr>
    <tr><td>Int4-AWQ</td><td>72.8</td><td>-</td><td>78.2</td><td>-</td></tr>
  </tbody>
  <tbody>
    <tr><td rowspan="4">Hunyuan-1.8B-Instruct</td>
    <td>BF16</td>          <td>63.4</td><td>56.7</td><td>76.7</td><td>47.2</td></tr>
    <tr><td>FP8-Static</td><td>62.5</td><td>55.2</td><td>75.1</td><td>47.7</td></tr>
    <tr><td>Int4-GPTQ</td><td>60.9</td><td>-</td><td>73.0</td><td>44.4</td></tr>
    <tr><td>Int4-AWQ</td><td>61.7</td><td>-</td><td>71.7</td><td>43.6</td></tr>
  </tbody>
  <tbody>
    <tr><td rowspan="4">Hunyuan-0.5B-Instruct</td>
    <td>BF16</td>          <td>29.6</td><td>17.2</td><td>52.8</td><td>23.3</td></tr>
    <tr><td>FP8-Static</td><td>29.6</td><td>17.2</td><td>51.6</td><td>22.5</td></tr>
    <tr><td>Int4-GPTQ</td><td>26.8</td><td>-</td><td>50.9</td><td>23.3</td></tr>
    <tr><td>Int4-AWQ</td><td>26.3</td><td>-</td><td>48.9</td><td>23.3</td></tr>
  </tbody>
</table>


#### Qwen3系列模型

Qwen3系列模型的`BF16`、`FP8-Static`、`FP8-Dynamic`、`INT8-Dynamic`、`INT4-GPTQ`、`INT4-AWQ`在`CEVAL`、`MMLU`、`GSM8K`、`HUMANEVAL`上的评测结果如下：

<table>
  <thead>
    <tr><th>Model</th><th>Quantization</th><th>CEVAL</th><th>MMLU</th><th>GSM8K</th><th>HUMANEVAL</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="4">Qwen3-0.6B</td><td>BF16</td><td>45.84</td><td>47.21</td><td>42.99</td><td>19.51</td></tr>
    <tr><td>FP8-Static</td><td>45.99</td><td>46.87</td><td>38.06</td><td>18.90</td></tr>
    <tr><td>FP8-Dynamic</td><td>45.99</td><td>46.93</td><td>38.29</td><td>20.73</td></tr>
    <tr><td>INT8-Dynamic</td><td>45.17</td><td>46.95</td><td>41.17</td><td>21.34</td></tr>
    <tr><td rowspan="6">Qwen3-8B</td><td>BF16</td><td>79.27</td><td>74.78</td><td>87.79</td><td>63.41</td></tr>
    <tr><td>FP8-Static</td><td>78.23</td><td>74.79</td><td>86.96</td><td>62.20</td></tr>
    <tr><td>FP8-Dynamic</td><td>78.45</td><td>74.75</td><td>87.64</td><td>62.80</td></tr>
    <tr><td>INT8-Dynamic</td><td>78.01</td><td>74.84</td><td>86.96</td><td>67.07</td></tr>
    <tr><td>INT4-GPTQ</td><td>77.19</td><td>73.26</td><td>86.43</td><td>62.20</td></tr>
    <tr><td>INT4-AWQ</td><td>76.15</td><td>73.59</td><td>86.96</td><td>63.41</td></tr>
    <tr><td rowspan="6">Qwen3-14B</td><td>BF16</td><td>83.06</td><td>78.90</td><td>88.40</td><td>55.49</td></tr>
    <tr><td>FP8-Static</td><td>82.62</td><td>78.57</td><td>89.46</td><td>57.32</td></tr>
    <tr><td>FP8-Dynamic</td><td>82.24</td><td>78.92</td><td>88.32</td><td>52.44</td></tr>
    <tr><td>INT8-Dynamic</td><td>81.87</td><td>78.13</td><td>86.28</td><td>56.10</td></tr>
    <tr><td>INT4-GPTQ</td><td>81.05</td><td>78.02</td><td>87.34</td><td>57.93</td></tr>
    <tr><td>INT4-AWQ</td><td>82.02</td><td>77.68</td><td>84.23</td><td>61.59</td></tr>
    <tr><td rowspan="5">Qwen3-32B</td><td>BF16</td><td>86.55</td><td>82.00</td><td>74.53</td><td>37.80</td></tr>
    <tr><td>FP8-Static</td><td>86.92</td><td>81.78</td><td>70.20</td><td>39.63</td></tr>
    <tr><td>FP8-Dynamic</td><td>86.55</td><td>81.89</td><td>70.43</td><td>38.41</td></tr>
    <tr><td>INT4-GPTQ</td><td>86.18</td><td>81.01</td><td>-</td><td>43.29</td></tr>
    <tr><td>INT4-AWQ</td><td>86.18</td><td>81.54</td><td>-</td><td>36.59</td></tr>
    <tr><td rowspan="4">Qwen3-30B-A3B</td><td>BF16</td><td>83.66</td><td>79.36</td><td>89.99</td><td>31.71</td></tr>
    <tr><td>FP8-Static</td><td>83.95</td><td>79.47</td><td>89.01</td><td>31.10</td></tr>
    <tr><td>FP8-Dynamic</td><td>84.10</td><td>79.40</td><td>89.16</td><td>32.93</td></tr>
    <tr><td>INT8-Dynamic</td><td>83.36</td><td>79.48</td><td>89.16</td><td>34.15</td></tr>
    <tr><td rowspan="4">Qwen3-235B-A22B</td><td>BF16</td><td>89.60</td><td>86.28</td><td>85.29</td><td>27.44</td></tr>
    <tr><td>FP8-Static</td><td>89.67</td><td>86.19</td><td>86.96</td><td>27.44</td></tr>
    <tr><td>FP8-Dynamic</td><td>89.67</td><td>86.18</td><td>85.22</td><td>28.05</td></tr>
    <tr><td>INT8-Dynamic</td><td>88.93</td><td>86.20</td><td>86.20</td><td>23.78</td></tr>
    <tr><td rowspan="5">QwQ-32B</td><td>BF16</td><td>85.74</td><td>82.03</td><td>73.31</td><td>42.68</td></tr>
    <tr><td>FP8-Static</td><td>85.44</td><td>81.91</td><td>75.36</td><td>42.68</td></tr>
    <tr><td>FP8-Dynamic</td><td>85.07</td><td>81.93</td><td>75.66</td><td>42.07</td></tr>
    <tr><td>INT4-GPTQ</td><td>84.03</td><td>81.26</td><td>68.23</td><td>45.73</td></tr>
    <tr><td>INT4-AWQ</td><td>83.58</td><td>81.01</td><td>68.69</td><td>43.29</td></tr>
  </tbody>
</table>

#### Qwen2.5VL系列模型

Qwen2.5VL系列模型的`BF16`、`FP8-Static`、`FP8-Dynamic`、`INT4-GPTQ`、`INT4-AWQ`在`MMMU_VAL`、`DocVQA_VAL`、`ChartQA_TEST`上的评测结果如下：

<table>
  <thead>
    <tr><th>Model</th><th>Quantization</th><th>MMMU_VAL</th><th>MMLDocVQA_VALU</th><th>ChartQA_TEST</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="5">Qwen2.5VL-3B</td><td>BF16</td><td>47.11</td><td>78.57</td><td>80.32</td></tr>
    <tr><td>FP8-Static</td><td>47.33</td><td>79.34</td><td>79.68</td></tr>
    <tr><td>FP8-Dynamic</td><td>45.99</td><td>46.93</td><td>38.29</td></tr>
    <tr><td>INT4-GPTQ</td><td>46.56</td><td>77.20</td><td>78.96</td></tr>
    <tr><td>INT4-AWQ</td><td>45.78</td><td>-</td><td>79.60</td></tr>
   <tr><td rowspan="5">Qwen2.5VL-7B</td><td>BF16</td><td>45.44</td><td>89.71</td><td>84.64</td></tr>
    <tr><td>FP8-Static</td><td>47.00</td><td>89.83</td><td>85.92</td></tr>
    <tr><td>FP8-Dynamic</td><td>47.22</td><td>89.80</td><td>88.64</td></tr>
    <tr><td>INT4-GPTQ</td><td>46.67</td><td>90.45</td><td>-</td></tr>
    <tr><td>INT4-AWQ</td><td>45.67</td><td>89.28</td><td>-</td></tr>
    <tr><td rowspan="5">Qwen2.5VL-32B</td><td>BF16</td><td>57.00</td><td>90.03</td><td>-</td></tr>
    <tr><td>FP8-Static</td><td>57.00</td><td>89.88</td><td>-</td></tr>
    <tr><td>FP8-Dynamic</td><td>56.44</td><td>89.88</td><td>-</td></tr>
    <tr><td>INT4-GPTQ</td><td>55.22</td><td>89.80 </td><td>-</td></tr>
    <tr><td>INT4-AWQ</td><td>55.22</td><td>90.30</td><td>-</td></tr>
    <tr><td rowspan="5">Qwen2.5VL-72B</td><td>BF16</td><td>58.78</td><td>94.39</td><td>85.60</td></tr>
    <tr><td>FP8-Static</td><td>57.89</td><td>94.41</td><td>85.84</td></tr>
    <tr><td>FP8-Dynamic</td><td>58.67</td><td>94.38</td><td>85.60</td></tr>
    <tr><td>INT4-GPTQ</td><td>57.56</td><td>94.46</td><td>86.48</td></tr>
    <tr><td>INT4-AWQ</td><td>58.78</td><td>94.19</td><td>87.28</td></tr>
  </tbody>
</table>

#### DeepSeek系列模型

DeepSeek-R1-0528模型的`FP8-Block-Wise`、`W4A8-FP8`在`GPQA Diamond`、`AIME 2024`、`SimpleQA`、`LiveCodeBench`上的评测结果如下：

<table>
  <thead>
    <tr><th>Model</th><th>Quantization</th><th>GPQA Diamond</th><th>AIME 2024</th><th>SimpleQA</th><th>LiveCodeBench</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="6">DeepSeek-R1-0528</td><td>FP8-Block-Wise</td><td>78.28</td><td>88.67</td><td>27.8</td><td>77.1</td></tr>
    <tr><td>W4A8-FP8</td><td>77.37</td><td>88.67</td><td>26.83</td><td>78.86</td></tr>
  </tbody>
</table>

> **备注**：
> - 以上评测结果使用TRT-LLM框架部署测试5次求平均
> - 评测时使用的超参如下:
> ```json
>{
>  "top_k": 20,
>  "top_p": 0.6,
>  "temperature": 0.7,
>  "output_seq_len": 32768,
>  "max_input_seq_len": 16384
>}
>```


#### 其他模型

其他模型的`BF16`、`FP8-Static`、`FP8-Dynamic`、`INT4-GPTQ`、`INT4-AWQ`在`CEVAL`、`MMLU`、`GSM8K`上的评测结果如下：

<table>
  <thead>
    <tr><th>Model</th><th>Quantization</th><th>CEVAL</th><th>MMLU</th><th>GSM8K</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="3">Qwen2.5-1.5B-Instruct</td><td>BF16</td><td>67.01</td><td>60.05</td><td>54.28</td></tr>
    <tr><td>FP8-Static</td><td>66.27</td><td>60.23</td><td>-</td></tr>
    <tr><td>FP8-Dynamic</td><td>66.79</td><td>60.08</td><td>51.71</td></tr>
    <tr><td rowspan="5">Qwen2.5-7B-Instruct</td><td>BF16</td><td>81.20</td><td>74.55</td><td>79.98</td></tr>
    <tr><td>FP8-Static</td><td>81.13</td><td>74.03</td><td>79.30</td></tr>
    <tr><td>FP8-Dynamic</td><td>80.31</td><td>74.07</td><td>79.00</td></tr>
    <tr><td>INT4-GPTQ</td><td>79.05</td><td>73.05</td><td>74.75</td></tr>
    <tr><td>INT4-AWQ</td><td>79.35</td><td>73.22</td><td>79.38</td></tr>
    <tr><td rowspan="5">Qwen2.5-32B-Instruct</td><td>BF16</td><td>87.30</td><td>83.21</td><td>81.73</td></tr>
    <tr><td>FP8-Static</td><td>87.59</td><td>83.08</td><td>81.58</td></tr>
    <tr><td>FP8-Dynamic</td><td>87.30</td><td>83.04</td><td>81.58</td></tr>
    <tr><td>INT4-GPTQ</td><td>86.70</td><td>82.45</td><td>82.03</td></tr>
    <tr><td>INT4-AWQ</td><td>87.00</td><td>82.64</td><td>-</td></tr>
    <tr><td rowspan="5">DeepSeek-R1-Distill-Qwen-7B</td><td>BF16</td><td>53.49</td><td>53.80</td><td>75.74</td></tr>
    <tr><td>FP8-Static</td><td>53.57</td><td>54.17</td><td>76.19</td></tr>
    <tr><td>FP8-Dynamic</td><td>52.97</td><td>54.13</td><td>74.15</td></tr>
    <tr><td>INT4-GPTQ</td><td>51.86</td><td>52.44</td><td>75.89</td></tr>
    <tr><td>INT4-AWQ</td><td>53.49</td><td>53.70</td><td>-</td></tr>
    <tr><td rowspan="5">DeepSeek-R1-Distill-Qwen-14B</td><td>BF16</td><td>77.71</td><td>74.28</td><td>85.67</td></tr>
    <tr><td>FP8-Static</td><td>77.56</td><td>74.66</td><td>86.73</td></tr>
    <tr><td>FP8-Dynamic</td><td>76.82</td><td>74.63</td><td>87.11</td></tr>
    <tr><td>INT4-GPTQ</td><td>74.29</td><td>72.37</td><td>84.61</td></tr>
    <tr><td>INT4-AWQ</td><td>74.81</td><td>73.00</td><td>86.05</td></tr>
    <tr><td rowspan="5">DeepSeek-R1-Distill-Qwen-32B</td><td>BF16</td><td>84.18</td><td>80.89</td><td>87.41</td></tr>
    <tr><td>FP8-Static</td><td>83.43</td><td>80.90</td><td>87.57</td></tr>
    <tr><td>FP8-Dynamic</td><td>83.73</td><td>81.10</td><td>86.43</td></tr>
    <tr><td>INT4-GPTQ</td><td>84.10</td><td>79.80</td><td>86.73</td></tr>
    <tr><td>INT4-AWQ</td><td>82.84</td><td>80.15</td><td>87.19</td></tr>
  </tbody>
</table>

### （2）投机采样
#### Qwen3 Series Models
Qwen3系列的Eagle3模型在MT-bench/HunmanEval/GSM8K/Alpaca上的加速结果如下：

<table>
  <thead>
    <tr>
        <th>&nbsp</th><th>&nbsp</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">MT-bench</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">HumanEval</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">GSM8K</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">Alpaca</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">Mean</th></tr>
    <tr><th>Temperature</th><th>Model</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th></tr>
  </thead>
  <tbody>
    <!-- <tr><td colspan="12" style="text-align: center; vertical-align: middle;"><strong>Temperature=0</strong></td></tr> -->
    <tr><td rowspan="6"><strong>T=0</strong></td>
    <td>Qwen3-1.7B</td><td>2.05x</td><td>2.81</td><td>2.07x</td><td>2.93</td><td>2.11x</td><td>2.98</td><td>1.93x</td><td>2.69</td><td>2.04x</td><td>2.85</td></tr>
    <tr> <td>Qwen3-4B</td><td>2.21x</td><td>3.01</td><td>2.36x</td><td>3.24</td><td>2.42x</td><td>3.13</td><td>2.32x</td><td>2.75</td><td>2.33x</td><td>3.03</td></tr>
    <tr><td>Qwen3-8B</td><td>2.63x</td><td>3.65</td><td>2.76x</td><td>3.85</td><td>2.82x</td><td>3.90</td><td>2.62x</td><td>3.48</td><td>2.70x</td><td>3.72</td></tr>
    <tr><td>Qwen3-14B</td><td>2.23x</td><td>3.30</td><td>2.53x</td><td>3.74</td><td>2.56x</td><td>3.79</td><td>2.16x</td><td>3.13</td><td>2.37x</td><td>3.49</td></tr>
    <tr><td>Qwen3-32B</td><td>2.39x</td><td>2.78</td><td>2.37x</td><td>2.81</td><td>2.47x</td><td>2.92</td><td>2.42x</td><td>2.53</td><td>2.41x</td><td>2.76</td></tr>
    <tr><td>Qwen3-30B-A3B</td><td>2.84x</td><td>3.63</td><td>2.27x</td><td>3.09</td><td>2.64x</td><td>3.42</td><td>2.83x</td><td>3.56</td><td>2.64x</td><td>3.42</td></tr>
    <!-- <tr><td colspan="12" style="text-align: center; vertical-align: middle;"><strong>Temperature=1</strong></td></tr> -->
    <tr><td rowspan="6"><strong>T=1</strong></td>
    <td>Qwen3-1.7B</td><td>1.74x</td><td>2.53</td><td>1.86x</td><td>2.70</td><td>1.82x</td><td>2.69</td><td>1.72x</td><td>2.46</td><td>1.93x</td><td>2.60</td></tr>
    <tr><td>Qwen3-4B</td><td>1.93x</td><td>2.60</td><td>2.00x</td><td>2.84</td><td>2.11x</td><td>2.82</td><td>2.34x</td><td>2.50</td><td>1.75x</td><td>2.69</td></tr>
    <tr><td>Qwen3-8B</td><td>1.98x</td><td>2.75</td><td>2.25x</td><td>3.11</td><td>2.31x</td><td>3.15</td><td>2.10x</td><td>2.76</td><td>2.90x</td><td>2.94</td></tr>
    <tr><td>Qwen3-14B</td><td>1.71x</td><td>2.61</td><td>1.95x</td><td>2.87</td><td>2.04x</td><td>3.08</td><td>1.68x</td><td>2.55</td><td>2.90x</td><td>2.78</td></tr>
    <tr><td>Qwen3-32B</td><td>1.62x</td><td>1.91</td><td>1.71x</td><td>2.05</td><td>1.78x</td><td>2.10</td><td>1.80x</td><td>1.95</td><td>1.62x</td><td>2.00</td></tr>
    <tr><td>Qwen3-30B-A3B</td><td>1.91x</td><td>2.46</td><td>2.00x</td><td>2.64</td><td>1.90x</td><td>2.53</td><td>1.80x</td><td>2.32</td><td>1.90x</td><td>2.48</td></tr>
  </tbody>
</table>

Hunyuan系列的Eagle3模型在MT-bench/HunmanEval/GSM8K/Alpaca上的加速结果如下：

<table>
  <thead>
    <tr>
        <th>&nbsp</th><th>&nbsp</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">MT-bench</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">HumanEval</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">GSM8K</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">Alpaca</th>
        <th colspan="2" style="text-align: center; vertical-align: middle;">Mean</th></tr>
    <tr><th>Temperature</th><th>Model</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th><th>Speedup</th><th>τ</th></tr>
  </thead>
  <tbody>
    <!-- <tr><td colspan="12" style="text-align: center; vertical-align: middle;"><strong>Temperature=0</strong></td></tr> -->
    <tr><td rowspan="3"><strong>T=0</strong></td>
    <td>Hunyuan-1.8B-Instruct</td><td>1.97x</td><td>2.90</td><td>2.58x</td><td>3.73</td><td>2.61x</td><td>3.71</td><td>1.71x</td><td>2.43</td><td>2.22x</td><td>3.19</td></tr>
    <tr> <td>Hunyuan-4B-Instruct</td><td>1.77x</td><td>2.60</td><td>2.64x</td><td>3.35</td><td>2.14x</td><td>3.17</td><td>1.72x</td><td>2.57</td><td>2.07x</td><td>2.92</td></tr>
    <tr><td>Hunyuan-7B-Instruct</td><td>2.22x</td><td>3.58</td><td>3.59x</td><td>5.47</td><td>2.96x</td><td>4.68</td><td>1.64x</td><td>2.56</td><td>2.60x</td><td>4.07</td></tr>
    <!-- <tr><td colspan="12" style="text-align: center; vertical-align: middle;"><strong>Temperature=1</strong></td></tr> -->
    <tr><td rowspan="3"><strong>T=1</strong></td>
    <td>Hunyuan-1.8B-Instruct</td><td>1.58x</td><td>2.36</td><td>2.35x</td><td>3.56</td><td>2.23x</td><td>3.38</td><td>1.26x</td><td>1.87</td><td>1.86x</td><td>2.79</td></tr>
    <tr><td>Hunyuan-4B-Instruct</td><td>1.36x</td><td>2.05</td><td>1.97x</td><td>2.86</td><td>1.72x</td><td>2.68</td><td>1.14x</td><td>1.76</td><td>1.55x</td><td>2.34</td></tr>
    <tr><td>Hunyuan-7B-Instruct</td><td>1.90x</td><td>3.11</td><td>3.12x</td><td>5.09</td><td>2.74x</td><td>4.34</td><td>1.47x</td><td>2.39</td><td>2.31x</td><td>3.73</td></tr>
  </tbody>
</table>

## 📝许可协议
本项目的代码依照 [License for AngelSlim](LICENSE) 协议开源。

## 🔗引用
```
@software{AngelSlim2025,
    title={{AngelSlim}},
    author={Tencent AngelSlim Project Contributors},
    year={2025},
    month={7},
    url={https://github.com/Tencent/AngelSlim},
}
```

## 💬技术交流

- AngelSlim正在快速迭代更新中，后续会推出更多的功能，有问题或建议欢迎通过[GitHub Issues](https://github.com/Tencent/AngelSlim/issues)给我们提issue，或者加入[微信技术交流群](./docs/source/assets/angel_slim_wechat.png)。
