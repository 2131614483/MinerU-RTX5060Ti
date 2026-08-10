---
tags:
  - MinerU
  - GPU适配
  - CUDA
  - 显卡兼容矩阵
  - 环境配置
  - claude
date: 2026-08-10
---

# MinerU × 显卡 × CUDA 配置组合总表

> 生成日期：2026-08-10 ｜ 数据来源：**PyPI（mineru 各版本依赖声明）+ download.pytorch.org（torch 官方 wheel 索引）+ PyTorch / onnxruntime 官方发布说明**（均为当日实测抓取）
> 相关：[[MinerU-RTX5060Ti-适配报告-详细]] ｜ [[MinerU-RTX5060Ti-适配报告-简略]]

## 0. 三条铁律（先读这个）

1. **MinerU 从不锁定 CUDA 构建。** 实测 PyPI：MinerU 2.0.0 → 3.4.4（及 4.0.0 alpha）的所有版本，pipeline/vlm 依赖都只声明 `torch>=2.6.0,<3`（早期 2.x 是 `>=2.2.2,<3`）。GPU 能不能用、用哪个 CUDA，**完全由你实际装的 torch 决定**——这是"CUDA 版本由 torch 决定"的来源（`docs/zh/reference/changelog.md` 1.3.0 条目）。
2. **判断依据是 torch 轮子的 CUDA 构建版本（cu 后缀），不是 torch 版本号。** 例如 `torch 2.13.0+cu126` 一样跑不了 RTX 50 系，因为 CUDA 12.6 编译器不认识 sm_120。
3. **RTX 50 系（Blackwell，sm_120）必须 cu128/cu129/cu130 构建的 torch（torch ≥ 2.7.0）。** 40 系及更早：cu124/cu126 默认构建即可。默认 `pip install torch`（Windows/PyPI）装的是 cu126 → **在 50 系上必翻车**（`is_available()=False` 或 `no kernel image`）。

## 1. 显卡架构 ↔ 计算能力 ↔ 最低 CUDA（覆盖全系列）

| 架构 | 计算能力 | 消费级显卡 | 数据中心显卡 | 该架构最低 CUDA | torch 怎么选 |
|---|---|---|---|---|---|
| Blackwell（消费） | 12.0 (sm_120) | **RTX 50 系**：5050 / 5060 / 5060 Ti / 5070 / 5070 Ti / 5080 / 5090 / 5090D（含笔记本） | — | **12.8** | **必须 cu128/cu129/cu130** |
| Blackwell（数据中心） | 10.0/10.3 (sm_100/103) | RTX PRO 5000 Blackwell | **B100 / B200 / B300 / GB200 / GB300** | 12.8 | cu128（2.7.0 起）+，推荐 cu129/cu130 |
| Hopper | 9.0 (sm_90) | — | H100 / H200 / H20 / H800 | 11.8 | cu118 起，推荐 cu126/cu128 |
| Ada Lovelace | 8.9 (sm_89) | **RTX 40 系**：4050~4090（含笔记本） | L4 / L40 / L40S / RTX 6000 Ada | 11.8 | cu124 / cu126 |
| Ampere（消费） | 8.6 (sm_86) | **RTX 30 系**：3050~3090 | A10 / A40 / A16 | 11.1 | cu118 / cu121 / cu124 / cu126 |
| Ampere（数据中心） | 8.0 (sm_80) | — | A100 / A30 | 11.0 | cu118 起 |
| Turing | 7.5 (sm_75) | **RTX 20 系**、GTX 16 系 | T4 / Quadro RTX | 10.0 | cu118 / cu124 / cu126 |
| Volta | 7.0 (sm_70) | Titan V | V100 | 9.0 | cu118 |
| Pascal | 6.0/6.1 (sm_60/61) | **GTX 10 系**：1050~1080 | P100 / P40 | 8.0 | cu124（torch 2.6.0）/ cu126（≤2.7.1）。**勿装 cu128+ 的 torch 2.8** |
| Maxwell | 5.0/5.2 (sm_50/52) | GTX 9 系 / GTX 750 Ti | M40 | 7.0 | 同上（cu126 及更老构建） |

> ⚠️ PyTorch 2.8.0 发布说明：**cu128 / cu129 构建已移除 Maxwell 和 Pascal（sm_50~sm_60）支持**（二进制体积限制）。所以老卡（GTX 900/10 系）只能停在 cu126 及更老构建。

## 2. torch 官方 CUDA 构建矩阵（2026-08-10 实测 download.pytorch.org）

| CUDA 构建 | 可用 torch 版本 | 含 Maxwell/Pascal (sm50-60) | 含 Turing~Ampere (sm70-89) | 含 Hopper (sm90) | 含 Blackwell (sm100/120) | 备注 |
|---|---|---|---|---|---|---|
| cu118 | 2.0.0 – 2.7.1 | ✅ | ✅ | ✅ | ❌ | 老机器兜底 |
| cu121 | 2.1.0 – 2.5.1 | ✅ | ✅ | ✅ | ❌ | |
| cu124 | 2.4.0 – 2.6.0 | ✅ | ✅ | ✅ | ❌ | GTX 10 系可用的最新之一 |
| **cu126** | **2.6.0 – 2.13.0** | ✅（≤2.7.1 稳妥） | ✅ | ✅ | **❌** | **Windows 默认 pip 装的就是它 → 50 系翻车根源** |
| **cu128** | **2.7.0 – 2.11.0** | ❌（2.8 起移除） | ✅ | ✅ | **✅** | **RTX 50 系（sm_120）官方支持起点** |
| cu129 | 2.8.0 – 2.13.0 | ❌ | ✅ | ✅ | ✅ | 2.8 新增，含 sm_100（B200） |
| cu130 | 2.9.0 – 2.13.0 | ❌ | ✅ | ✅ | ✅ | 最新，CUDA 13.0 |

配套 torchvision（同 CUDA 源）：torch 2.6↔tv 0.21、2.7↔0.22、2.8↔0.23、2.9↔0.24、2.10↔0.25、2.11↔0.26、2.12↔0.27、2.13↔0.28（实测索引一致）。

## 3. MinerU 各版本依赖要求（2026-08-10 实测 PyPI）

| MinerU | pipeline torch 要求 | vlm torch 要求 | onnxruntime | transformers | VLM 推理框架 |
|---|---|---|---|---|---|
| 2.0.0 – 2.1.x | `>=2.2.2,<3`（排除 2.5.x） | `>=2.6.0` | 未声明 | `>=4.49` | sglang |
| 2.2.0 | `>=2.6.0,<2.8.0` | `>=2.6.0` | `>1.17.0` | `>=4.49` | — |
| 2.5.x | `>=2.6.0,<2.8.0` | `>=2.6.0` | `>1.17.0` | `>=4.49` | vllm 0.10.1.1 |
| 2.6.x / 2.7.x | `>=2.6.0,<3` | `>=2.6.0` | `>1.17.0` | `>=4.51.1` | vllm<0.12（Linux）/ lmdeploy（Win） |
| 3.0.0 – 3.4.4 | `>=2.6.0,<3` | `>=2.6.0` | `>1.17.0` | `>=4.57.3` | vllm<0.22（Linux）/ lmdeploy<0.12（Win） |
| 4.0.0a5（preview） | basic: `>=2.6.0,<3` | — | `>=1.17.0`（基础依赖） | `>=4.57.3` | 同上 |

要点：onnxruntime 自 2.2 起进入 pipeline 依赖；paddle/paddleocr 在 1.3 系列已被 `paddleocr2torch` 完全替代（无 paddle 冲突）；VLM 后端 Windows 走 lmdeploy、Linux 走 vLLM。

## 4. 综合配置组合表：按显卡选 MinerU + torch

| 你的显卡 | 架构/CC | 手动装 torch（关键步骤） | 可用 MinerU | 默认 pip 能跑？ |
|---|---|---|---|---|
| RTX 5090/5080/5070/5070 Ti/5060 Ti/5060/5050 | sm_120 | **cu128**（torch 2.7.0–2.11.0）或 cu129/cu130 | 任意（推荐 3.x） | ❌ 必须手动 cu128 源 |
| B200/B100/B300/GB200 | sm_100/103 | cu128（2.7.0+）/ cu129 / cu130 | 任意 | ❌ |
| H100/H200 | sm_90 | cu126 或 cu128 | 任意 | ✅ |
| RTX 4090/4080/4070/4060 系列 | sm_89 | cu124 / cu126 | 任意 | ✅ |
| RTX 3090/3080/3070/3060/3050 | sm_86 | cu118 / cu124 / cu126 | 任意 | ✅ |
| A100/A30 | sm_80 | cu118 起 | 任意 | ✅ |
| RTX 2080/2070/2060、T4、GTX 1660 | sm_75 | cu126（≤2.7.1）或 cu124（2.6.0） | 任意 | ✅ |
| V100 | sm_70 | cu118 | 任意 | ✅ |
| GTX 1080/1070/1060/1050 | sm_61 | cu124（2.6.0）/ cu126（≤2.7.1） | 任意 | ✅（但别升 torch 2.8 cu128） |
| GTX 980/GTX 750 Ti | sm_50/52 | cu126 及更老 | 任意 | ✅（同上） |

## 5. Windows 专项说明

- **驱动门槛**（近似，以 NVIDIA 发行说明为准）：cu118≥450 / cu121≥530 / cu124≥550 / cu126≥560 / cu128≥570 / cu130≥580。RTX 50 系最低约 570；本机 610.47 远超。
- **默认 pip 陷阱**：Windows 上 `pip install torch`（或让 MinerU 依赖自动解析）拿到 cu126 → 50 系 GPU 用不了。50 系必须显式：
  ```bash
  pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
  ```
- **VLM 加速（可选）**：Blackwell 上官方 FAQ 推荐 `lmdeploy 0.11.1+cu128` Windows wheel（`docs/zh/faq/index.md`）；MinerU 3.4.4 要求 `lmdeploy>=0.10.2,<0.12`，0.11.1 符合。Linux 上 vLLM（50 系需 vLLM≥0.10.1.1 + cu128 环境）。**不用 VLM 时纯 torch（transformers 后端）同样可行**——本机 RTX 5060 Ti 即此路线。
- **表格识别 onnxruntime-gpu**：MinerU 里 onnxruntime 只服务表格模型；50 系表格上 GPU 需 `onnxruntime-gpu≥1.22`（CUDA 12.8 EP，v1.22 起 GPU 包全线 CUDA 12.x）。装 CPU-only 的 onnxruntime 时表格优雅回退 CPU，不影响整体。
- **显存要求**：MinerU ≥1.3 优化后最低约 **6GB** 显存可跑 pipeline；16GB（如 5060 Ti）可全流程 + VLM。
- **国产卡**：MinerU 2.6+ 适配昇腾/沐曦/海光/燧原/摩尔线程/寒武纪/昆仑芯/太初/壁仞等（NPU 走 torch_npu，非 CUDA 体系，不适用本矩阵）。

## 6. 推荐当前组合（2026-08）

| 场景 | 组合 |
|---|---|
| RTX 50 系（含 5060 Ti） | MinerU 3.4.4 + **torch 2.7.1+cu128** + torchvision 0.22.1+cu128 + torchaudio 2.7.1+cu128；onnxruntime CPU 即可（要表格 GPU 换 ≥1.22 onnxruntime-gpu） |
| RTX 40 / 30 / 20 系 | MinerU 3.4.4 + 默认 cu126 torch（≥2.6.0） |
| GTX 10 系 / 9 系（老卡） | MinerU 3.x + 手动 pin `torch==2.6.0+cu124`（或 `2.7.1+cu126`），**不要升 torch 2.8 cu128** |
| 验证命令 | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"` → `True (12, 0)` 即 Blackwell 正确配置 |

## 7. 数据来源（均可复核）

- mineru 各版本 `requires_dist`：`https://pypi.org/pypi/mineru/<version>/json`
- torch/torchvision 各 CUDA 构建可用版本：`https://download.pytorch.org/whl/cu128/torch/`（cu118/cu121/cu124/cu126/cu129/cu130 同理）
- PyTorch 2.7.0 发布说明：Blackwell 支持（sm100/sm120）加入；2.8.0：cu128/cu129 移除 sm50-60
- onnxruntime v1.22.0 发布说明：GPU 包要求 CUDA 12.x；1.25.0：CUDA 最低 12.0
- MinerU 本地：`docs/zh/faq/index.md`、`docs/zh/reference/changelog.md`（1.3.0 的 CUDA 兼容说明）、`pyproject.toml`
