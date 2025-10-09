English | [简体中文](README.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/source/assets/logos/angelslim_logo_light.png">
    <img alt="AngelSlim" src="./docs/source/assets/logos/angelslim_logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
Dedicated to building a more intuitive, comprehensive, and efficient LLMs compression toolkit.
</h3>

<p align="center">
          📖 <a href="https://angelslim.readthedocs.io/">Documentation</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/AngelSlim">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/AngelSlim">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="./docs/source/assets/angel_slim_wechat.png">WeChat</a> | &nbsp&nbsp🫨 <a href="https://discord.com/invite/dHVNeuNdFt">Discord</a>
<br>
</p>

## Table of Contents

- [Latest Updates](#latest-updates)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [How to Use](#how-to-use)
  - [Install AngelSlim](#install-angelslim)
  - [Quick Start](#quick-start)
  - [deployment & Evaluation](#deployment)
- [Benchmark](#benchmark)
- [License](#license)
- [Citation](#citation)
- [Technical Discussion](#technical-discussion)

## 📣Latest Updates
- [25/09/30] We now open-source the implementation of the reasoning early-exit algorithm: *SpecExit: Accelerating Large Reasoning Model via Speculative Exit* | [[Paper]](http://arxiv.org/abs/2509.24248) | [[Docs]](https://angelslim.readthedocs.io/zh-cn/latest/features/speculative_decoding/spec_exit.html)
- [25/09/26] We now release the the ternary quantization *TEQUILA: TRAPPING-FREE TERNARY QUANTIZATION FOR LARGE LANGUAGE MODELS* | [[Paper]](https://arxiv.org/abs/2509.23809) | [[Code]](https://github.com/Tencent/AngelSlim/tree/tequila/TernaryQuant)
- [25/09/24] We now support the PTQ quantification of NVFP4 for the Qwen3 series models. We also opensource [Qwen3-32B-NVFP4](https://huggingface.co/AngelSlim/Qwen3-32B_nvfp4) and [Qwen3-235B-A22B-NVFP4](https://huggingface.co/AngelSlim/Qwen3-235B-A22B_nvfp4) weights.
- [25/09/01] We now support ​FP8 quantization​ of the [Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B-fp8) translation model. And enabled ​Torch inference and Benchmark evaluation​ for Eagle3. And implemented support for ​quantization and Cache​ for [FLUX](https://github.com/Tencent/AngelSlim/tree/main/configs/flux). And support ​quantization​ for the [Seed-OSS](https://github.com/Tencent/AngelSlim/tree/main/configs/seed_oss).
- [25/08/06] We now support quantization for `Hunyuan 0.5B/1.8B/4B/7B` and multimodal model `Qwen2.5VL 3B/7B/32B/72B`, including `FP8/INT4` algorithms, and quantization for `DeepSeek-R1/V3` and `Kimi-K2`, including `FP8-Static` and `W4A8-FP8` algorithms. We also opensource `Hunyuan 1.8B/4B/7B` series Eagle3 model weight.
- [25/07/04] We now support quantization for `Hunyuan/Qwen2.5/Qwen3/DeepSeek-R1-Distill-Qwen` and other models, including `INT8/FP8/INT4` algorithms. We also opensource `Qwen3` series Eagle3 model weight.

Coming soon:
- [ ] Diffusion model compression support.
- [ ] Release of new algorithm for speculative sampling.

## 🌟Key Features

- **Highly Integrated**: This toolkit integrates mainstream compression algorithms into a unified framework, offering developers one-click access with exceptional ease of use.
- **Continuous Innovation**: Beyond integrating widely-used industry algorithms, we are continuously researching better compression algorithms, which will be gradually open-sourced in the future.
- **Performance-Driven**: We continuously optimize end-to-end performance in model compression workflows and algorithm deployment, such as enabling quantization of models like Qwen3-235B and DeepSeek-R1 on a single GPU.

## 💼Supported Models

### Quantization
Currently supports the following LLMs, including Hunyuan-Dense, Hunyuan-MoE, Qwen3-Dense, Qwen3-MoE, Qwen2.5, DeepSeek-R1 distilled Qwen models, and QwQ::

| Model | FP8-Dynamic | FP8-Static | INT8-Dynamic | INT4-GPTQ | INT4-AWQ |
| --------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------- | ------------ | --------- | -------- |
| [Hunyuan-Dense](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)                                                         | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Hunyuan-MoE](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be)                             | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Qwen3-Dense](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)                            | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Qwen3-MoE](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)                              | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Qwen2.5](https://huggingface.co/collections/AngelSlim/qwen2-25-quant-68652d6cbdf5c0d4b1c4499a)                             | ✅           | ✅          | ✅            | ✅         | ✅        |
| [DeepSeek-R1-Distill-Qwen](https://huggingface.co/collections/AngelSlim/deepseek-r1-distill-quant-68652f16a9c206b030b05f7f) | ✅           | ✅          | ✅            | ✅         | ✅        |
| [QwQ](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)                                    | ✅           | ✅          | ✅            | ✅         | ✅        |

### Speculative Decoding

#### Eagle3
The Eagle3 weights for the Qwen3 series model are now available.

| Qwen3  Models   | Hunyuan Models     |
| ----------|----------|
| ✅ [Qwen3-1.7B](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3)    |✅ [Hunyuan-1.8B-Instruct](https://huggingface.co/AngelSlim/Hunyuan-1.8B-Instruct_eagle3)    |
| ✅ [Qwen3-4B](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)        |✅ [Hunyuan-4B-Instruct](https://huggingface.co/AngelSlim/Hunyuan-4B-Instruct_eagle3)        |
| ✅ [Qwen3-8B](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)        |✅ [Hunyuan-7B-Instruct](https://huggingface.co/AngelSlim/Hunyuan-7B-Instruct_eagle3)        |
| ✅ [Qwen3-14B](https://huggingface.co/AngelSlim/Qwen3-14B_eagle3)      |
| ✅ [Qwen3-32B](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3)      |
| ✅ [Qwen3-30B-A3B](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3)  |

## 🛎️How to Use

### Install AngelSlim

We recommend using `pip` to install the latest stable version of `AngelSlim`:

```shell
pip install angelslim
```

Alternatively, you can clone the repository and install from source in editable mode:

```shell
cd AngelSlim && python setup.py install
```

For more detailed installation instructions, please refer to the [Installation Documentation](https://angelslim.readthedocs.io/zh-cn/latest/getting_started/installation.html).

### Quick Start

#### Quantization

After installing `AngelSlim`, you can quickly start by running the following script to perform static `FP8` quantization on the `Qwen3-1.7B` model:

* One-click Start

  ```shell
  python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-1_7b_fp8_static.yaml
  ```

  This example will load the HuggingFace model and perform activation value calibration using the `dataset` specified in the config file, saving the quantized model weights.

* Code-based Start

  To perform dynamic `FP8` quantization on `Qwen3-1.7B`:

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

For more details, please refer to the [Quick Start Documentation](https://angelslim.readthedocs.io/zh-cn/latest/getting_started/quickstrat.html).

#### Speculative_Decoding

##### Eagle3 PyTorch Performance Testing

After installing `AngelSlim`, you can quickly start Eagle3 PyTorch performance testing with the following script:

```bash
python3 tools/spec_benchmark.py \
    --base-model-path /path/to/base/model \
    --eagle-model-path /path/to/eagle/model \
    --model-id your_model_id \
    --mode both
```

For more details, please refer to the [Quick Start Documentation](https://angelslim.readthedocs.io/zh-cn/latest/getting_started/quickstrat.html). 

### Deployment and Testing

### 1. Offline Inference

If you need to load a quantized model via `transformers`, please set the `deploy_backend: huggingface` in the `global` configuration before quantizing the model, or manually modify the `ignored_layers` field in the `config.json` file located in the quantized model output directory to `ignore`.

To test offline inference with a quantized model loaded via `transformers`, run the following command:

```shell
python deploy/offline.py $MODEL_PATH
```

Where `MODEL_PATH` is the path to the quantized model output.

#### 2. API Service Deployment

After specifying the quantized model path `MODEL_PATH`, you can deploy an OpenAI-compatible API service using the following LLMs inference frameworks:

**vLLM**

Use the following script to launch a [vLLM](https://github.com/vllm-project/vllm) server, recommended version `vllm>=0.8.5.post1`. For MOE INT8 quantized models, vllm>=0.9.0 is required.


```shell
bash deploy/run_vllm.sh $MODEL_PATH
```

**SGLang**


Use the following script to launch a [SGLang](https://github.com/sgl-project/sglang) server, recommended version `sglang>=0.4.6.post1`.

```shell
bash deploy/run_sglang.sh $MODEL_PATH
```

#### 3. Service Invocation

Invoke requests via [OpenAI's API format](https://platform.openai.com/docs/api-reference/introduction):

```shell
bash deploy/openai.sh $MODEL_PATH
```

#### 4. Performance Evaluation

Evaluate the performance of quantized model using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), recommended version`lm-eval>=0.4.8`:

```shell
bash deploy/lm_eval.sh $MODEL_PATH
```

For more detaileds, please refer to the [Deployment Documentation](https://angelslim.readthedocs.io/zh-cn/latest/deployment/deploy.html).


## 📈 Benchmark

### (1) Quantization

The performance test results for selected models are shown below. For the complete benchmark, refer to the [Benchmark documentation](https://angelslim.readthedocs.io/zh-cn/latest/performance/quantization/benchmarks.html)

#### Hunyuan Series Models

Benchmark results for the `Hunyuan-Instruct` model with `FP8`, `INT4-AWQ` and `INT4-GPTQ` quantization algorithms on datasets including`OlympiadBench`, `AIME 2024` and `DROP`:

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

#### Qwen3 Series Models

Benchmark results for Qwen3 series models with `FP8-Static`, `FP8-Dynamic`, `INT4-GPTQ`, and `INT4-AWQ` quantization algorithms on datasets including `CEVAL`, `MMLU`, `GSM8K`, and `HUMANEVAL`:

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

#### Qwen2.5VL Series Models

Benchmark results for Qwen2.5VL series models with `BF16`、`FP8-Static`、`FP8-Dynamic`、`INT4-GPTQ`、`INT4-AWQ` quantization algorithms on datasets including `MMMU_VAL`、`DocVQA_VAL` and `ChartQA_TEST`：

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

#### DeepSeek Series Models

Benchmark results for DeepSeek-R1-0528 series models with `FP8-Block-Wise` and `W4A8-FP8` quantization algorithms on datasets including `GPQA Diamond`、`AIME 2024`、`SimpleQA` and `LiveCodeBench`：

<table>
  <thead>
    <tr><th>Model</th><th>Quantization</th><th>GPQA Diamond</th><th>AIME 2024</th><th>SimpleQA</th><th>LiveCodeBench</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="6">DeepSeek-R1-0528</td><td>FP8-Block-Wise</td><td>78.28</td><td>88.67</td><td>27.8</td><td>77.1</td></tr>
    <tr><td>W4A8-FP8</td><td>77.37</td><td>88.67</td><td>26.83</td><td>78.86</td></tr>
  </tbody>
</table>

> **Note**：
> - The above results are based on the average of 5 test runs deployed with TRT-LLM
> - The hyperparameters used during evaluation are as follows:
> ```json
>{
>  "top_k": 20,
>  "top_p": 0.6,
>  "temperature": 0.7,
>  "output_seq_len": 32768,
>  "max_input_seq_len": 16384
>}
>```

#### Other Models

Benchmark results for other models with `FP8-Static`, `FP8-Dynamic`, `INT4-GPTQ`, and `INT4-AWQ` quantization algorithms on datasets including `CEVAL`, `MMLU` and `GSM8K`:

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

### (2) Speculative Decoding

#### Qwen3 Series Models
Benchmark results for Qwen3 series models with `Eagle3` speculative decoding algorithm on datasets including `MT-bench`, `HunmanEval`, `GSM8K`, and `Alpaca`:

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

#### Hunyuan Series Models
Benchmark results for Hunyuan series models with `Eagle3` speculative decoding algorithm on datasets including `MT-bench`, `HunmanEval`, `GSM8K`, and `Alpaca`:

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

## 📝 License

The code for this project is open-sourced under the [License for AngelSlim](LICENSE).

## 🔗 Citation

```
@software{AngelSlim2025,
    title={{AngelSlim}},
    author={Tencent AngelSlim Project Contributors},
    year={2025},
    month={6},
    url={https://github.com/Tencent/AngelSlim},
}
```

## 💬 Technical Discussion

* AngelSlim is continuously iterating and new features will be released soon. If you have any questions or suggestions, please open an issue on [GitHub Issues](https://github.com/Tencent/AngelSlim/issues) or join our [WeChat technical discussion group](./docs/source/assets/angel_slim_wechat.png).
