---
tags:
  - MinerU
  - GPU适配
  - RTX5060Ti
  - CUDA
  - 环境配置
  - claude
date: 2026-08-10
---

# MinerU 适配 RTX 5060 Ti 研究报告（详细版）

> 研究日期：2026-08-10 ｜ 研究对象：`D:\MinerU-master`（MinerU 3.4.4，源码安装）
> 简略版：[[MinerU-RTX5060Ti-适配报告-简略]]

## 结论（TL;DR）

这套配置能在 RTX 5060 Ti 上跑通，**唯一核心原因是虚拟环境里显式安装了 PyTorch 的 CUDA 12.8（cu128）构建**——`torch==2.7.1+cu128`（含配套 `torchvision 0.22.1+cu128`、`torchaudio 2.7.1+cu128`）。

官方 git 安装方式会让 pip 解析到默认的 **cu126（CUDA 12.6）** 构建。RTX 5060 Ti 是 Blackwell 架构（sm_120），只有 CUDA ≥ 12.8 才带它的内核，所以 GPU 用不起来——这就是"网上 git 配置不适配 5060 Ti"的根源。

## 一、硬件与系统背景

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB（Blackwell / GB206） |
| 架构 | sm_120（compute capability **12.0**） |
| 驱动 | 610.47（CUDA UMD 13.3） |
| 系统 | Windows 10 Pro 22H2 |
| CUDA Toolkit | v12.8（`nvcc 12.8.93`，`CUDA_PATH=V12.8`） |
| Python | 3.12.10 |

## 二、RTX 5060 Ti 的硬性门槛

**sm_120（compute capability 12.0）是 CUDA 12.8 才引入的架构。** 低于 12.8 的 PyTorch 构建（cu121 / cu124 / cu126）编译目标里**没有** sm_120 内核。后果有两种：

- `torch.cuda.is_available()` 返回 `False`，MinerU 静默回退 CPU；
- 或强行调用时报 `no kernel image is available for execution on the device`。

无论哪种，都等于"GPU 用不上"。

## 三、官方配置为什么失败

`D:\MinerU-master\pyproject.toml` 里官方只声明：

- `pipeline` extra：`torch>=2.6.0,<3`、`torchvision`、`onnxruntime>1.17.0`（`pyproject.toml:98-103`）
- `vlm` extra：`torch>=2.6.0,<3`（`pyproject.toml:74-78`）

**没有锁定 CUDA 构建版本**。按 README 的安装命令（`README_zh-CN.md:307`）：

```bash
uv pip install -e .[all] -i https://mirrors.aliyun.com/pypi/simple
```

从 PyPI/阿里云镜像解析 → Windows 默认解析到 **cu126** 的 torch 2.6/2.7 → 不含 sm_120 → 不适配 5060 Ti。这正是官方 FAQ 里"Windows 直接安装后推理速度很慢"那一节描述的场景。

官方 FAQ（`docs/zh/faq/index.md:14-22`）给 Blackwell 用户的标准答案是：

```powershell
$env:LMDEPLOY_VERSION = "0.11.1"
$env:PYTHON_VERSION = "312"
$wheel = "https://github.com/InternLM/lmdeploy/releases/download/v$($env:LMDEPLOY_VERSION)/lmdeploy-$($env:LMDEPLOY_VERSION)+cu128-cp$($env:PYTHON_VERSION)-cp$($env:PYTHON_VERSION)-win_amd64.whl"
pip install $wheel --extra-index-url https://download.pytorch.org/whl/cu128
```

即靠 **cu128 的 torch + lmdeploy** 走 VLM 路线。

## 四、本环境实际安装的软件（实测证据）

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | **2.7.1+cu128** | 显式 cu128 构建（`+cu128` 后缀只存在于 `download.pytorch.org/whl/cu128` 发布） |
| torchvision | 0.22.1+cu128 | 与 torch 匹配 |
| torchaudio | 2.7.1+cu128 | 与 torch 匹配 |
| onnxruntime | 1.28.0 | **CPU-only**，仅表格识别使用 |
| lmdeploy | 0.6.5 | 旧版，本环境未实际依赖（非 FAQ 的 cu128 0.11.1） |
| transformers | 4.57.6 | |
| numpy | 1.26.4 | |
| mineru | 3.4.4 | 源码 editable 安装于 `D:\MinerU-master` |

**实测探针结果**（用 `.venv` 的 python 运行，未改动任何文件）：

- `torch.cuda.is_available()` = **True**
- `torch.cuda.get_device_name(0)` = `NVIDIA GeForce RTX 5060 Ti`
- `torch.cuda.get_device_capability(0)` = **(12, 0)** → 确认是 sm_120
- GPU 矩阵乘（1024×1024 matmul）**实测通过**
- `onnxruntime.get_available_providers()` = `['AzureExecutionProvider', 'CPUExecutionProvider']` → **无 CUDAExecutionProvider**

## 五、为什么这套配置能在 GPU 上跑通

1. **设备选择自动落到 CUDA**：环境里**没有设置** `MINERU_DEVICE_MODE`，`mineru/utils/config_reader.py:105` 的 `get_device()` 逻辑为 `MINERU_DEVICE_MODE` 未设时，`torch.cuda.is_available()` 为 True 就返回 `"cuda"`。
2. **pipeline 后端里所有 torch 模型全部走 GPU**：
   - 版面：`pp_doclayoutv2.py`（PaddlePaddle DocLayout V2 的 PyTorch 实现）
   - OCR：`pytorch_paddle.py`（PaddleOCR 的 PyTorch 实现）
   - 公式：Unimernet（torch）
3. **onnxruntime 只在表格识别用**（`mineru/model/table/rec/`、`cls/`），本环境是 CPU-only 包，`onnxruntime_provider.py:47-50` 在 `CUDAExecutionProvider` 不可用时**优雅回退 CPU provider**。所以表格在 CPU 上跑，**不影响整体 GPU 加速，也不是 5060 Ti 卡住的点**。

## 六、结论归纳

这套配置成功 = 三个条件同时成立：

1. **torch/torchvision/torchaudio 用 cu128 构建**——这是 Blackwell 上唯一能让 PyTorch 正常工作的打开方式；
2. **驱动足够新**——RTX 50 系列最低要求约 570（CUDA 12.8），本机 610.47 远超；
3. **不强制设备模式**——未设 `MINERU_DEVICE_MODE=cpu`，让 `get_device()` 自动选 `cuda`。

## 七、复现方法（在别处做对）

```bash
# 1) 显式装 cu128 构建（关键一步，别让 pip 解析默认 cu126）
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 2) 装 MinerU（core 即可，表格若要 GPU 另装 onnxruntime-gpu）
pip install -e .[core]

# 3) 验证
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

## 八、FAQ / 备注

- **表格加速**：本环境 `onnxruntime 1.28.0` 为 CPU-only，表格识别跑 CPU；如需表格也上 GPU，需另装 `onnxruntime-gpu`（注意其 CUDA 12.8 EP 版本）。
- **VLM 路线**：官方 Blackwell 推荐 `lmdeploy 0.11.1+cu128` wheel；本环境走的是**纯 cu128 torch 路线**（lmdeploy 0.6.5 存在但未依赖），同样成功，两条路都成立。
- 研究过程全程**只读**：未改动 `D:\MinerU-master` 下任何文件。

> 全型号/全版本配置组合：[[MinerU-版本-显卡-CUDA-配置矩阵]]
