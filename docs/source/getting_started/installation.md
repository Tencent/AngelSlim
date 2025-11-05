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