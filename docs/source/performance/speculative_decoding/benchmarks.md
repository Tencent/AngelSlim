# 投机采样Benchmark

## Eagle3

### 1. Qwen3 Series Models

| Model | Method | GSM8K |      | Alpaca |      | HumanEval |      | MT-bench |      | Mean |      |
| :---  | :---   | :---  | :--- | :---   | :--- | :---      | :--- | :---     | :--- | :--- | :--- |
|       |        | **throughput (tokens/s)** | **accept length** | **throughput (tokens/s)** | **accept length** | **throughput (tokens/s)** | **accept length** | **throughput (tokens/s)** | **accept length** | **throughput (tokens/s)** | **accept length** |
| **Qwen3-1.7B** | Vanilla | 376.42 | 1 | 378.86 | 1 | 378.38 | 1 | 390.53 | 1 | 381.05 | 1 |
|                | Eagle3 | 616.9 | 2.13 | 653.29 | 2.19 | 680.1 | 2.2 | 621.44 | 2.17 | 642.93 | 2.17 |
| **Qwen3-4B** | Vanilla | 229.05 | 1 | 235.29 | 1 | 234.66 | 1 | 234.04 | 1 | 233.26 | 1 |
|              | Eagle3 | 389.35 | 2.07 | 395.97 | 2.1 | 377.84 | 2.08 | 384.6 | 2.07 | 386.94 | 2.08 |
| **Qwen3-8B** | Vanilla | 149.63 | 1 | 149.93 | 1 | 153.85 | 1 | 153.81 | 1 | 151.81 | 1 |
|              | Eagle3 | 257.32 | 2 | 266.69 | 2.02 | 244.89 | 1.97 | 258.2 | 1.97 | 257.52 | 1.99 |
| **Qwen3-14B** | Vanilla | 92.97 | 1 | 92.66 | 1 | 92.94 | 1 | 94.46 | 1 | 93.26 | 1 |
|               | Eagle3 | 153.72 | 1.87 | 140.46 | 1.78 | 144.68 | 1.76 | 142.45 | 1.74 | 145.33 | 1.79 |
| **Qwen3-32B**     | Vanilla | 43.39  | 1    | 43.38  | 1    | 43.19  | 1     | 43.3   | 1    | 43.32  | 1    |
|                   | Eagle3  | 80.43  | 2.01 | 72.49  | 1.9  | 71.57  | 1.86  | 74.1   | 1.86 | 74.1   | 1.91 |
| **Qwen3-30B-A3B** | Vanilla | 311.84 | 1    | 320.43 | 1 | 325.77 | 1 | 325.42 | 1 | 320.87 | 1 |
|                   | Eagle3  | 453.97 | 2.1  | 432.45 | 2.04 | 428.81 | 2.02  | 437.06 | 2.01 | 438.07 | 2.04 |

### 2. VLM Models
#### 2.1 Qwen3-VL Series Models

<table><thead>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th colspan="2">GSM8K</th>
    <th colspan="2">Alpaca</th>
    <th colspan="2">HumanEval</th>
    <th colspan="2">MT-bench</th>
    <th colspan="2">MATH-500</th>
    <th colspan="2">MMMU</th>
    <th colspan="2">MMStar</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
  </tr>
  <tr>
    <td rowspan="2">Qwen3-VL-2B-Instruct</td>
    <td>Vanilla</td>
    <td>348.55</td>
    <td>1</td>
    <td>350.9</td>
    <td>1</td>
    <td>346.07</td>
    <td>1</td>
    <td>346.31</td>
    <td>1</td>
    <td>82.96</td>
    <td>1</td>
    <td>83.27</td>
    <td>1</td>
    <td>81.63</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Eagle3</td>
    <td>511.52</td>
    <td>2.11</td>
    <td>560.55</td>
    <td>2.26</td>
    <td>826.01</td>
    <td>3.39</td>
    <td>555.22</td>
    <td>2.29</td>
    <td>163.09</td>
    <td>2.57</td>
    <td>154.18</td>
    <td>2.55</td>
    <td>139.73</td>
    <td>2.31</td>
  </tr>
  <tr>
    <td rowspan="2">Qwen3-VL-4B-Instruct</td>
    <td>Vanilla</td>
    <td>212.87</td>
    <td>1</td>
    <td>213.24</td>
    <td>1</td>
    <td>211.69</td>
    <td>1</td>
    <td>212.1</td>
    <td>1</td>
    <td>67.96</td>
    <td>1</td>
    <td>65.88</td>
    <td>1</td>
    <td>67.75</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Eagle3</td>
    <td>415.29</td>
    <td>2.57</td>
    <td>372.89</td>
    <td>2.26</td>
    <td>459.37</td>
    <td>2.82</td>
    <td>382.33</td>
    <td>2.34</td>
    <td>141.87</td>
    <td>2.72</td>
    <td>104.44</td>
    <td>2.05</td>
    <td>107.07</td>
    <td>2.1</td>
  </tr>
  <tr>
    <td rowspan="2">Qwen3-VL-30B-A3B-Instruct</td>
    <td>Vanilla</td>
    <td>179.94</td>
    <td>1</td>
    <td>184.6</td>
    <td>1</td>
    <td>168.68</td>
    <td>1</td>
    <td>180.57</td>
    <td>1</td>
    <td>31.08</td>
    <td>1</td>
    <td>31.51</td>
    <td>1</td>
    <td>30.93</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Eagle3</td>
    <td>281.93</td>
    <td>2.82</td>
    <td>241.42</td>
    <td>2.13</td>
    <td>223.05</td>
    <td>2.57</td>
    <td>240.47</td>
    <td>2.19</td>
    <td>75.31</td>
    <td>2.79</td>
    <td>48.47</td>
    <td>1.78</td>
    <td>52.57</td>
    <td>1.94</td>
  </tr>
</tbody></table>

#### 2.2 HunyuanOCR Model

<table><thead>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th colspan="2">OmniDocBench</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
  </tr>
  <tr>
    <td rowspan="2">Hunyuan-OCR</td>
    <td>Vanilla</td>
    <td>70.12</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Eagle3</td>
    <td>108.1</td>
    <td>2.08</td>
  </tr>
</tbody>
</table>

### 3. Audio Models

#### 3.1 Qwen2-Audio Model

<table><thead>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th colspan="2">LibriSpeech</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
  </tr>
  <tr>
    <td rowspan="2">Qwen2_Audio</td>
    <td>Vanilla</td>
    <td>78.76</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Eagle3</td>
    <td>146.66</td>
    <td>3.51</td>
  </tr>
</tbody>
</table>

#### 3.2 Fun-CosyVoice3 Model

<table><thead>
  <tr>
    <th>Model</th>
    <th>Method</th>
    <th colspan="2">LibriTTS</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td></td>
    <td>throughput (tokens/s)</td>
    <td>accept length</td>
  </tr>
  <tr>
    <td rowspan="2">Fun-CosyVoice3</td>
    <td>Vanilla</td>
    <td>-</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Eagle3</td>
    <td>-</td>
    <td>1.96</td>
  </tr>
</tbody>
</table>