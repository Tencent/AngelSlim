(installation)=

# 安装教程

AngelSlim支持如下安装方式：

- [pip安装（推荐）](#pip安装（推荐）)
- [编译安装](#编译安装)
- [指定环境变量](#指定环境变量)

## pip安装（推荐）

#### 默认安装（LLM）

通过`pip`安装最新AngelSlim稳定发布版：

```shell
pip install angelslim
```

如果已经安装`AngelSlim`，通过下面的指令强制获取最新更新：

```shell
pip install --upgrade --force-reinstall --no-cache-dir angelslim
```

#### 投机采样安装

```shell
pip install angelslim[speculative]
```

#### 多模态安装

```shell
pip install angelslim[multimodal]
```

#### Diffusion安装

```shell
pip install angelslim[diffusion]
```

#### 全部安装

```shell
pip install angelslim[all]
```


:::{note}
- 如果pip安装失败，请检查联网是否正确，并更新pip：`pip install --upgrade pip`
- CUDA工具包: 可以参考[CUDA Toolkit 安装文档](https://developer.nvidia.com/cuda-toolkit-archive)安装所需要的版本；
- 与CUDA驱动程序的PyTorch版本：`AngelSlim`正确运行需要`torch>=2.4.1`，可以根据安装的 CUDA 驱动程序版本安装对应的[PyTorch 最新版本](https://pytorch.org/get-started/locally/)，或者所需要的[其他 PyTorch 版本](https://pytorch.org/get-started/previous-versions/)。
:::

## 编译安装

如果对工具代码做过改动，或者想使用main分支最新功能，推荐使用编译安装方式：

```shell
cd AngelSlim
python setup.py install
```

## 指定环境变量

如果对源码做了修改，更简易的方式是指定PYTHONPATH环境变量，例如：
```shell
export PYTHONPATH=Your/Path/to/AngelSlim/:$PYTHONPATH
```

:::{note}
指定环境变量后，需要和执行压缩算法的脚本在同一终端执行，比如放在同一个shell脚本内，先export PYTHONPATH环境变量，然后运行压缩程序代码。
:::

## Windows Installation (with FP8 Triton Support)

AngelSlim supports Windows with FP8 Triton kernels. Follow these steps to build from source:

```batch
:: Clone the repository
git clone https://github.com/Tencent/AngelSlim.git
cd AngelSlim

:: Create and activate virtual environment (Python 3.10 recommended)
uv venv --python 3.10
.venv\Scripts\activate

:: Install base dependencies
uv pip install packaging wheel setuptools ninja numpy==1.26.4 pip build psutil

:: Install PyTorch with CUDA 12.8 support
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

:: Install Triton for Windows
uv pip install -U triton-windows

:: Configure Visual Studio build environment
set INCLUDE=
set LIB=
set LIBPATH=
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Configure CUDA environment
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_HOME%\bin;%PATH%
set DISTUTILS_USE_SDK=1

:: Set target CUDA architectures (adjust based on your GPU)
set TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0

:: Build the wheel
set DG_USE_LOCAL_VERSION=0
python setup.py bdist_wheel

:: Verify FP8 Triton kernels are working
python -c "import torch; from angelslim.compressor.diffusion.kernels.python.quantizers import fp8_per_block_quant_triton; from angelslim.compressor.diffusion.kernels.python.gemm import fp8_gemm_triton_block; a,b=torch.randn(128,256,device='cuda'),torch.randn(512,256,device='cuda'); aq,a_s=fp8_per_block_quant_triton(a); bq,b_s=fp8_per_block_quant_triton(b); c=fp8_gemm_triton_block(aq,a_s,bq,b_s); print(f'FP8 GEMM OK: {c.shape}, {c.dtype}')"
```

**Requirements:**
- Windows 10/11 with NVIDIA GPU (Ampere or newer recommended)
- Visual Studio 2022 with C++ build tools
- CUDA Toolkit 12.8
- Python 3.10

**Environment Variables:**
- `ANGELSLIM_BACKEND`: Force backend selection (`triton` or `pytorch`)
- `ANGELSLIM_TORCH_COMPILE`: Enable/disable torch.compile (`0` or `1`)