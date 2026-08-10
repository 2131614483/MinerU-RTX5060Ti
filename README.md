# MinerU-RTX5060Ti

MinerU 适配 RTX 5060 Ti（及全部显卡×CUDA）配置研究。

## 核心结论

RTX 50 系（Blackwell，sm_120）跑 MinerU 成功的关键：**显式安装 PyTorch 的 cu128（CUDA 12.8）构建**（`torch>=2.7.0+cu128`）。官方 git 安装会让 pip 解析到默认 cu126 构建，不含 Blackwell 内核，因此 GPU 用不起来。

MinerU 本身（2.x–3.x）从不锁定 CUDA 构建，只声明 `torch>=2.6.0,<3`——CUDA 版本完全由你装的 torch 决定。

## 文档

| 文档 | 内容 |
|---|---|
| [适配报告-详细](MinerU-RTX5060Ti-适配报告-详细.md) | RTX 5060 Ti 适配完整研究：证据链、官方失败原因、本机实测、复现方法 |
| [适配报告-简略](MinerU-RTX5060Ti-适配报告-简略.md) | 一句话结论 + 复现命令的快速参考卡 |
| [版本-显卡-CUDA-配置矩阵](MinerU-版本-显卡-CUDA-配置矩阵.md) | MinerU 各版本 × 全系列显卡 × CUDA 配置组合总表（2026-08 实测自 PyPI 与 PyTorch 官方索引） |

## 关键复现命令

```bash
# RTX 50 系必须显式装 cu128 构建
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# 验证
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# 期望输出：True (12, 0)
```

## 同步约定

本仓库与 Obsidian 笔记 `研究/MinerU-RTX5060Ti/` 一一对应，报告同步自该目录。
