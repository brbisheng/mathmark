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
┌─────────────────────────────────────────────┐
│ L6 验证链    C2PA 签名 + 哈希链 (取证)      │
│ L5 元数据    EXIF/XMP + 自定义字段          │
│ L4 语义水印  数学符号/例题/步骤签名 ★核心    │
│ L3 不可见    TrustMark + DWT-DCT-SVD + Cox  │
│ L2 可见      半透明logo + 对抗性扰动         │
│ L1 指纹      pHash (快速筛查)               │
└─────────────────────────────────────────────┘
```

每层各有不同攻击面，多层叠加使攻击者**必须同时攻破所有相关层**才能完全去除水印。

## 安装

```bash
# 基础版 (核心水印 + 图像处理)
pip install -e .

# 完整版 (含深度学习、OCR、GUI、报告)
pip install -e ".[all]"
```

## 快速开始

### 1. 初始化教师身份

```bash
mathmark init --name "王老师" --output ~/.mathmark/
```

### 2. 配置签名

```bash
mathmark sign --problem "x^2-5x+6=0" --variant "x^2-7x+12=0"
```

### 3. 添加水印

```bash
mathmark embed lesson1.png marked/ --layers all
```

### 4. 验证归属

```bash
mathmark verify stolen.png --report evidence.pdf
```

## 性能 (CPU 1024×1024)

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
