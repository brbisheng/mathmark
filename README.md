# MathMark

> 多层抗AI去水印工具，专为数学教学内容版权保护设计

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

## 为什么需要 MathMark

2024-2026 年 SOTA AI 去水印技术（NeurIPS 2024 挑战赛冠军方案）能以 **95.7%** 成功率剥离主流单一图像水印。传统水印方案在以下攻击下表现脆弱：

- 干净噪声扩散再生（Diffusion-based regeneration）
- 视觉释义攻击（Caption-guided regen）
- GradCAM 局部模糊攻击
- 频域聚类攻击

**但是**：数学老师的内容具有**稳定可识别的教学风格**——符号习惯、变量命名、招牌例题、解题步骤。这些**无法被扩散重生成攻击去除**——攻击者必须重新解题才能改变风格，代价远高于经济收益。

**MathMark 的核心创新**：以**数学语义水印**（L4 层）为主防御层，传统图像水印为辅助层，构建 6 层防御栈。

## 6 层防御架构

```
┌─────────────────────────────────────────────────────────────┐
│ L6 验证链    C2PA 签名 + 哈希链 (取证)                      │
│ L5 元数据    EXIF (APP1) + XMP (APP1 for JPEG / iTXt for PNG)│
│ L4 语义水印  识别老师签名风格 + 建议变体 (不修改像素) ★核心  │
│ L3 不可见    TrustMark + DWT-DCT-SVD + Cox                  │
│ L2 可见      半透明 logo + 对抗性扰动 (multiply 混合, 不遮字)│
│ L1 指纹      pHash + dHash (快速筛查)                       │
└─────────────────────────────────────────────────────────────┘
```

每层各有不同攻击面，多层叠加使攻击者**必须同时攻破所有相关层**才能完全去除水印。

> **L3a/L3b/L6 降级说明**：当 `trustmark` / `invisible-watermark` / `c2pa-python` 未安装时，
> 对应层会**自动降级**到更简单的实现 (L3a → DCT 扩频, L3b → 仅 spread-spectrum,
> L6 → 简化 JSON manifest + EC/Ed25519 签名)。生产环境请 `pip install -e ".[all]"`。

## 安装

```bash
# 基础版 (核心水印 + 图像处理)
pip install -e .

# 完整版 (含深度学习、OCR、TrustMark、C2PA、报告)
pip install -e ".[all]"
```

## Colab 快速运行

在 Google Colab 里建议先跑最小闭环，确认安装、加水印、验证都能工作：

```bash
!git clone https://github.com/brbisheng/mathmark.git
%cd mathmark
!pip install -e .
!python3 sandbox/run.py demo
!mathmark --version
```

Colab 基础安装会使用轻量 fallback：没有真实 OCR 时 L4 会标记为 mock，没有 TrustMark/C2PA 标准库时 L3a/L6 会走降级实现。需要真实 OCR、TrustMark 或报告能力时再安装完整依赖：

```bash
!pip install -e ".[all]"
```

## 快速开始

### 0. 一键自检（最简单的方式确认 pipeline 正常）

```bash
python3 sandbox/run.py demo
```

会自动生成合成图 → 嵌入 6 层 → 验证，零配置。退出码 0 = 全部通过。

### 1. 初始化教师身份

```bash
mathmark init --name "王老师"
```

### 2. 配置签名

```bash
mathmark sign --problem "x^2-5x+6=0" --variant "x^2-7x+12=0"
mathmark sign --marker "∴" --marker "故"
```

### 3. 添加水印（单张）

```bash
# 注意: input / output 都是位置参数, 没有 --input 标志
mathmark embed lesson1.png marked.png
```

### 4. 批量水印（目录）

```bash
mathmark embed ./in_dir/ ./out_dir/ --batch
```

### 5. 验证归属

```bash
mathmark verify marked.png
mathmark verify marked.png --report evidence.pdf   # 证据 PDF (需 reportlab)
```

## CLI 子命令一览

| 子命令 | 用途 |
|---|---|
| `mathmark init` | 生成密钥对 + 默认配置 |
| `mathmark sign` | 登记招牌例题 / 变体 / 结论标记 |
| `mathmark embed` | 嵌入水印 (单图或 `--batch` 目录) |
| `mathmark verify` | 取证验证 (含 L1-L6 综合打分) |
| `mathmark extract` | 仅提取某一层证据 (调试用) |
| `mathmark benchmark` | 性能基准 (耗时 + 内存峰值) |
| `mathmark doctor` | 环境健康检查 (依赖 / 模型 / 配置) |

`embed` / `verify` / `extract` 的图像路径都是**位置参数** (第一个 = 输入, 第二个 = 输出)，没有 `--input` / `--output` 标志。

## 性能 (CPU 1024×1024)

> 以下时间基于 `[all]` 完整依赖 (trustmark / invisible-watermark / c2pa-python / paddleocr)。
> 基础安装 + L3a/L3b/L6 走降级路径时, L3a ≈ 100ms, L6 ≈ 5ms, 总计 <1.5s/图。

| 层级 | 方法 | 时间 |
|---|---|---|
| L1 | pHash | <20ms |
| L2 | 可见+扰动 | <200ms |
| L3a | TrustMark | ~800ms |
| L3b | DWT-DCT-SVD | ~1500ms |
| L3c | Cox | <50ms |
| L4 | 语义水印 | <300ms |
| L5 | 元数据 | <30ms |
| L6 | C2PA | <100ms |
| **总计** | **所有层** | **<3s/图** |

## 文档

- [架构说明](docs/architecture.md)
- [用户指南](docs/user-guide.md)
- [攻击抵抗测试](docs/attack_resistance.md)
- [诚实局限](docs/limitations.md)

## 许可

MIT License - 详见 [LICENSE](LICENSE)
