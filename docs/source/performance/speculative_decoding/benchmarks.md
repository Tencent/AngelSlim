# 投机采样Benchmark

## Eagle3

### Qwen3 Series Models

|                  |              | MT-bench         |            | HumanEval         |             | GSM8K          |         | Alpaca         |          | Mean          |        |
|------------------|--------------|------------------|------------|-------------------|-------------|----------------|---------|----------------|----------|---------------|--------|
|                  | Model        |  Speedup         |  τ         |  Speedup          |  τ          |  Speedup       |  τ      |  Speedup       |  τ       |  Speedup      |  τ     |
|                  | Qwen3-1.7B   | 2.05x            | 2.81       | 2.07x             | 2.93        | 2.11x          | 2.98    | 1.93x          | 2.69     | 2.04x         | 2.85   |
|                  | Qwen3-4B     | 2.21x            | 3.01       | 2.36x             | 3.24        | 2.42x          | 3.13    | 2.32x          | 2.75     | 2.33x         | 3.03   |
| **Temperature=0**| Qwen3-8B     | 2.63x            | 3.65       | 2.76x             | 3.85        | 2.82x          | 3.90    | 2.62x          | 3.48     | 2.70x         | 3.72   |
|                  | Qwen3-14B    | 2.23x            | 3.30       | 2.53x             | 3.74        | 2.56x          | 3.79    | 2.16x          | 3.13     | 2.37x         | 3.49   |
|                  | Qwen3-32B    | 2.39x            | 2.78       | 2.37x             | 2.81        | 2.47x          | 2.92    | 2.42x          | 2.53     | 2.41x         | 2.76   |
|                  | Qwen3-30B-A3B| 2.84x            | 3.63       | 2.27x             | 3.09        | 2.64x          | 3.42    | 2.83x          | 3.56     | 2.64x         | 3.42   |
|                  |              |                  |            |                   |             |                |         |                |          |               |        |
|                  | Qwen3-1.7B   | 1.74x            | 2.53       | 1.86x             | 2.70        | 1.82x          | 2.69    | 1.93x          | 2.46     | 1.75x         | 2.60   |
|                  | Qwen3-4B     | 1.93x            | 2.60       | 2.00x             | 2.84        | 2.11x          | 2.82    | 2.34x          | 2.50     | 1.75x         | 2.69   |
| **Temperature=1**| Qwen3-8B     | 1.98x            | 2.75       | 2.25x             | 3.11        | 2.31x          | 3.15    | 2.10x          | 2.76     | 2.90x         | 2.94   |
|                  | Qwen3-14B    | 1.71x            | 2.61       | 1.95x             | 2.87        | 2.04x          | 3.08    | 1.68x          | 2.55     | 2.90x         | 2.78   |
|                  | Qwen3-32B    | 1.62x            | 1.91       | 1.71x             | 2.05        | 1.78x          | 2.10    | 1.80x          | 1.95     | 1.62x         | 2.00   |
|                  | Qwen3-30B-A3B| 1.91x            | 2.46       | 2.00x             | 2.64        | 1.90x          | 2.53    | 1.80x          | 2.32     | 1.90x         | 2.48   |

### Hunyuan Series Models

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
    <tr><td rowspan="3"><strong>Temperature=0</strong></td>
    <td>Hunyuan-1.8B-Instruct</td><td>1.97x</td><td>2.90</td><td>2.58x</td><td>3.73</td><td>2.61x</td><td>3.71</td><td>1.71x</td><td>2.43</td><td>2.22x</td><td>3.19</td></tr>
    <tr> <td>Hunyuan-4B-Instruct</td><td>1.77x</td><td>2.60</td><td>2.64x</td><td>3.35</td><td>2.14x</td><td>3.17</td><td>1.72x</td><td>2.57</td><td>2.07x</td><td>2.92</td></tr>
    <tr><td>Hunyuan-7B-Instruct</td><td>2.22x</td><td>3.58</td><td>3.59x</td><td>5.47</td><td>2.96x</td><td>4.68</td><td>1.64x</td><td>2.56</td><td>2.60x</td><td>4.07</td></tr>
    <!-- <tr><td colspan="12" style="text-align: center; vertical-align: middle;"><strong>Temperature=1</strong></td></tr> -->
    <tr><td rowspan="3"><strong>Temperature=1</strong></td>
    <td>Hunyuan-1.8B-Instruct</td><td>1.58x</td><td>2.36</td><td>2.35x</td><td>3.56</td><td>2.23x</td><td>3.38</td><td>1.26x</td><td>1.87</td><td>1.86x</td><td>2.79</td></tr>
    <tr><td>Hunyuan-4B-Instruct</td><td>1.36x</td><td>2.05</td><td>1.97x</td><td>2.86</td><td>1.72x</td><td>2.68</td><td>1.14x</td><td>1.76</td><td>1.55x</td><td>2.34</td></tr>
    <tr><td>Hunyuan-7B-Instruct</td><td>1.90x</td><td>3.11</td><td>3.12x</td><td>5.09</td><td>2.74x</td><td>4.34</td><td>1.47x</td><td>2.39</td><td>2.31x</td><td>3.73</td></tr>
  </tbody>
</table>
</table>