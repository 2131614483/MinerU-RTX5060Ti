---
tags:
  - MinerU
  - GPU适配
  - RTX5060Ti
  - 环境配置
  - claude
date: 2026-08-10
---

# MinerU 适配 RTX 5060 Ti — 简略版

> 详细版（含全部证据链与复现方法）：[[MinerU-RTX5060Ti-适配报告-详细]]
> 研究日期：2026-08-10 ｜ 研究对象：`D:\MinerU-master`（MinerU 3.4.4）

## 一句话结论

**成功关键 = 显式安装 cu128（CUDA 12.8）构建的 torch。** 官方 git 装法会让 pip 解析到默认的 cu126（CUDA 12.6），而 RTX 5060 Ti（Blackwell，sm_120）必须 CUDA ≥ 12.8 才带内核，所以官方方式不适配。

## 核心要点

- 本环境装的是：`torch 2.7.1+cu128`、`torchvision 0.22.1+cu128`、`torchaudio 2.7.1+cu128`
- 实测：`torch.cuda.is_available()=True`，识别到 `RTX 5060 Ti`，capability **(12,0)**，GPU 矩阵乘通过
- 配套：驱动 610.47 + 系统 CUDA Toolkit 12.8（远超 50 系最低要求）
- MinerU 未设 `MINERU_DEVICE_MODE` → `get_device()` 自动返回 `cuda` → 版面 / OCR / 公式全走 GPU
- `onnxruntime 1.28.0` 是 **CPU-only**，只用于表格识别，缺 CUDA 时优雅回退 CPU——不是瓶颈

## 复现命令

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

然后正常 `pip install -e .[core]` 装 MinerU 即可。

## 踩坑提醒

RTX 50 系（Blackwell）一律要 **cu128 的 torch**；让 pip 默认解析（cu126/cu124）就是失败的根源，报错表现是 CUDA 用不上（`is_available()=False` 或 `no kernel image`）。

> GitHub 仓库：https://github.com/2131614483/MinerU-RTX5060Ti
