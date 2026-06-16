# MathMark 架构

## 概览

MathMark 采用 **6 层防御栈 (Defense-in-Depth)** 架构,以**数学语义水印 (L4)** 为核心创新层,传统图像水印为辅助层。设计目标是使攻击者**必须同时攻破多个独立层**才能完全去除水印。

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

## 设计哲学

### 为什么是 6 层

每一层针对不同的攻击面:

| 层级 | 攻击面 | 防御目标 |
|---|---|---|
| L1 指纹 | 简单复制/分享 | 快速检测重复 |
| L2 可见 | 人类直接裁切/裁剪 | 视觉威慑 + 抗 LaMa/MAT |
| L3 不可见 | JPEG 压缩/几何变换 | 像素域取证 |
| L4 语义 | **扩散重生成** | 抗 SOTA AI 攻击 |
| L5 元数据 | 平台剥离 EXIF | 局部平台有效 |
| L6 C2PA | 篡改 | 密码学证明 |

### 为什么 L4 是核心

数学内容具有**稳定可识别的教学风格**:

- 符号习惯 (`∴` vs `Q.E.D.`、`a/b/c` vs `α/β/γ`)
- 解题步骤 (`设...` 引入语、`化简得...` 过渡词)
- 招牌例题 (`x²-5x+6=0` 因式分解 = `(x-2)(x-3)`)
- 视觉记号 (箭头样式、圈重点)

这些**无法被扩散重生成攻击去除**——攻击者必须重新解题才能改变风格,代价远高于经济收益。

## 模块结构

```
mathmark/
├── core/                # 核心类型与配置
│   ├── types.py        # 全部数据类
│   ├── config.py       # YAML/JSON 配置加载
│   └── pipeline.py     # 6 层编排
├── layers/              # 6 层防御栈实现
│   ├── l1_fingerprint.py   # pHash
│   ├── l2_visible.py       # 可见水印 + 对抗扰动
│   ├── l3a_trustmark.py    # TrustMark (DCT 降级)
│   ├── l3b_dwt_dct_svd.py  # DWT-DCT-SVD
│   ├── l3c_cox_spread.py   # Cox 扩频
│   ├── l4_semantic.py      # 语义水印
│   ├── l5_metadata.py      # EXIF/XMP
│   └── l6_c2pa.py          # C2PA manifest
├── semantic/            # L4 子模块
│   ├── ocr.py          # OCR 引擎 (Tesseract/PaddleOCR/Mock)
│   ├── symbol_matcher.py   # 符号匹配
│   ├── step_matcher.py     # 步骤匹配
│   ├── example_db.py       # 招牌例题 hash 库
│   ├── injector.py        # 签名注入建议
│   └── recognizer.py      # 签名识别
├── crypto/              # 密码学
│   ├── keys.py         # 密钥管理 (EC/RSA/Ed25519)
│   └── hashing.py      # 哈希工具
├── verify/              # 取证工具
│   ├── extractor.py    # 多层水印提取
│   ├── consensus.py    # bit 一致性投票
│   └── report.py       # PDF/MD 报告生成
├── attacks/             # 攻击模拟器
│   ├── social_compress.py    # 微信/小红书/抖音
│   ├── local_blur.py         # 局部模糊/修复
│   └── diffusion_regen.py    # 扩散重生成
├── cli/main.py          # Typer CLI
└── utils/               # 通用工具
```

## 数据流

### 加水印流程

```
输入图像 (H×W×3 uint8)
    │
    ▼
[Layer 1] pHash 计算 (不修改图像)
    │
    ▼
[Layer 2] 可见水印 + 对抗扰动
    │  - render_visible_watermark()      ← diagonal_scatter (路透/法新风格)
    │  │    灰度 mask × 原图 (multiply 混合) 保护黑色数学内容不被遮挡
    │  - apply_adversarial_perturbation() ← 高频纹理扰动 (抗 LaMa/MAT 修复)
    ▼
[Layer 3a] TrustMark 嵌入 (DCT 降级)
[Layer 3b] DWT-DCT-SVD 嵌入
[Layer 3c] Cox 扩频嵌入
    │  - 每层都修改图像
    ▼
[Layer 4] 语义签名提取 (不修改图像)
    │  - OCR + 匹配符号/步骤/例题
    ▼
[Layer 5] EXIF/XMP 写入
    │  - 不修改像素,只更新 metadata
    ▼
[Layer 6] C2PA manifest 创建
    │  - 构造 signed manifest
    ▼
输出图像 + manifest.json
```

### 验证流程

```
输入可疑图像
    │
    ▼
[加载] 加载为 RGB numpy
    │
    ▼
[Layer 1] 重新计算 pHash, 与数据库比对
    │
    ▼
[Layer 4] OCR + 签名识别
    │  - 与配置的签名比对
    │  - 输出 similarity ∈ [0, 1]
    ▼
[Layer 3] 提取不可见水印 bits
    │  - 多方法 bit 投票
    ▼
[Layer 5] 读取 EXIF/XMP
    │
    ▼
[Layer 6] 验证 C2PA 签名 (用公钥)
    │
    ▼
[综合判定] 加权打分
    │  weights: L6(0.30) + L4(0.30) + L3(0.20) + L5(0.10) + L1(0.10)
    ▼
Verdict: STRONG_MATCH / PROBABLE_MATCH / WEAK / NO_MATCH
```

## 关键设计决策

### 1. 不可见水印三种方法的冗余

为什么同时用 TrustMark + DWT-DCT-SVD + Cox:

- **TrustMark (DCT 降级)**: 抗 JPEG 压缩强, 现代 SOTA
- **DWT-DCT-SVD**: 经典稳健方法, 抗几何变换
- **Cox 扩频**: 简单但抗噪声强

三者同时嵌入, 提取时投票, 互相补充。

### 2. 对角散布水印 + 全图对抗扰动

传统可见水印(角落 logo)5 秒裁掉 / DRAFT 大字挡公式 / 密铺像在图上写字 — 三种主流方案都不能兼顾"可见宣示"和"不破坏教学画面"。

MathMark 的 L2 用**路透/法新式对角散布**:
- 6 个 ~40px -30° 灰字散布在 3×2 网格 (砖墙式错开), 不可单边裁切抹除
- multiply 混合: 灰水印 × 白底 = 灰 (可见), 灰水印 × 黑字 = 黑 (数学内容不变)
- 用户可见的"宣示所有权" + 不挡公式

配合**全图均匀高频扰动**:
- 攻击者必须全局修复才能完全去除
- 即使部分去除,残留扰动也破坏修复的一致性
- 抗 GradCAM 局部模糊攻击

### 3. 签名注入 ≠ 自动改内容

L4 **不**自动修改原始教学内容(避免破坏教学逻辑)。

- 对于 PPT/文本源: 提供**可选**的自动注入模式
- 对于图片/手写: 只生成**建议清单**, 老师手动参考
- 签名配置可随时调整, 老师完全控制

### 4. CPU-only 的取舍

- 深度学习水印 (StegaStamp, Stable Signature) 需要 GPU → 排除
- 选择的方案: 经典方法 + 轻量 ONNX (TrustMark) + 自实现 Cox
- 性能预算: 1024×1024 图像 < 3 秒

## 性能特征

| 图像尺寸 | 6 层总耗时 | 主要瓶颈 |
|---|---|---|
| 256×256 | ~100ms | L3b DWT-DCT-SVD |
| 512×512 | ~250ms | L3b |
| 1024×1024 | ~3s | L3b |
| 2048×2048 | ~8s | L3b |

批处理可显著降低边际成本:
- 单图: ~250ms
- 8 批: ~2.3s/图

## 扩展点

- **L4 OCR 引擎**: 替换 `semantic/ocr.py` 中的实现
- **L3 不可见水印**: 添加新方法到 `layers/`
- **新攻击模拟**: 添加到 `attacks/`
- **新验证策略**: 修改 `verify/extractor.py` 的加权
