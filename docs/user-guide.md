# MathMark 用户指南

本指南面向数学老师,讲解如何用 MathMark 保护图文内容。

## 安装

```bash
# 基础版 (核心水印 + 图像处理)
pip install -e .

# 完整版 (含 GUI、PDF 报告、深度学习)
pip install -e ".[all]"
```

## 5 分钟快速开始

### 步骤 1: 初始化

```bash
mathmark init --name "王老师"
```

这会创建 `~/.mathmark/` 目录,包含:
- `config.yaml` - 全局配置
- `keys/private.pem` - 私钥 (本地, 不上传)
- `keys/public.pem` - 公钥
- `signatures/` - 签名配置目录

### 步骤 2: 配置签名

```bash
# 添加招牌例题
mathmark sign --problem "x^2-5x+6=0" --variant "x^2-7x+12=0"

# 添加结论标记
mathmark sign --marker "∴" --marker "故"

# 添加引入语
mathmark sign --marker "设" --marker "令"
```

### 步骤 3: 加水印

```bash
# 单张图
mathmark embed lesson1.png marked/lesson1.png

# 批量处理
mathmark embed ./lessons/ ./marked/ --batch

# 只用部分层
mathmark embed lesson.png marked.png --layers L1,L2,L4,L5,L6
```

### 步骤 4: 验证 (如果被偷了)

```bash
mathmark verify stolen.png --report evidence.pdf
```

会输出:
- 各层检测结果
- 综合判定 (STRONG_MATCH / PROBABLE_MATCH / WEAK)
- 可选的 PDF 法律证据报告

## 配置详解

### 签名配置 (`signatures/*.json`)

```json
{
  "teacher_id": "T2026-0001",
  "teacher_name": "王老师",
  "symbol": {
    "conclusion_markers": ["∴", "Q.E.D."],
    "variable_primary": ["x", "y", "z"],
    "set_notation": "∈",
    "vector_notation": "\\vec{AB}"
  },
  "step": {
    "introduction_phrases": ["设", "令", "记"],
    "transition_words": ["化简得", "整理得"],
    "conclusion_format": "故 {result}"
  },
  "example": {
    "signature_problems": [
      {
        "id": "default-001",
        "problem": "x^2 - 5x + 6 = 0",
        "expected_factoring": "(x-2)(x-3) = 0",
        "variants": ["x^2 - 7x + 12 = 0", "x^2 - 9x + 20 = 0"]
      }
    ]
  },
  "visual": {
    "arrow_style": "⟹",
    "underline_style": "wavy"
  }
}
```

### 全局配置 (`config.yaml`)

```yaml
teacher:
  id: T2026-0001
  name: 王老师
  public_key_path: ~/.mathmark/keys/public.pem
  private_key_path: ~/.mathmark/keys/private.pem

watermark:
  enabled_layers: all
  visible:
    text: "© 王老师数学课堂 2026"
    # 路透/法新风格: 6 个对角 -30° 灰字散布, 不可单边裁切抹除
    # 配合 multiply 混合: 灰水印 × 白底 = 灰 (可见), 灰水印 × 黑字 = 黑 (公式不被遮挡)
    position: diagonal_scatter  # diagonal_scatter | tiled | bottom-right | center
    opacity: 0.30               # multiply 模式下 = 视觉柔和度 (0.30 = 灰 ~48)
    font_size_ratio: 0.04       # 1024px 宽 → ~40px 字
    color: [160, 160, 160]      # 中浅灰
    scatter_count_x: 3          # 水平 3 个
    scatter_count_y: 2          # 垂直 2 个 (砖墙式错开 → 实际 6-7 个)
    scatter_angle: -30.0        # 倾斜角
    perturbation_strength: 0.02  # 对抗扰动强度 (抗 LaMa/MAT AI 修复)
  trustmark:
    model_path: models/trustmark.onnx
  dwt_dct_svd:
    alpha: 0.2
    block_size: 4
  cox_spread:
    strength: 0.05

verification:
  similarity_threshold: 0.5
  legal_export: true
```

## 工作流示例

### 场景 1: 备课 → 发布

```bash
# 1. 准备 PPT
mathmark sign --problem "x^2-5x+6=0" -c config.yaml

# 2. 截图
screenshot lesson.pptx → lesson_screenshot.png

# 3. 加水印
mathmark embed lesson_screenshot.png marked/lesson_screenshot.png

# 4. 发布到小红书/微信公众号
```

### 场景 2: 发现被抄袭

```bash
# 1. 保存抄袭图
wget stolen_image.png

# 2. 验证
mathmark verify stolen_image.png --report evidence.pdf

# 3. 查看报告
open evidence.pdf
```

如果判定为 STRONG_MATCH 或 PROBABLE_MATCH, 可以:
- 在平台提交版权投诉
- 用 PDF 报告作为法律证据
- 在签名清单中查询被偷的图

## 常见问题

### Q: 加水印会损坏图像吗?

A: 不会。L1-L4 都不修改像素, L2 和 L3 引入的扰动在视觉上不可察觉 (PSNR > 50dB)。

### Q: 性能如何?

A: 1024×1024 图像约 3 秒。批处理 8 张约 18 秒。可调整 `enabled_layers` 跳过不需要的层。

### Q: 支持视频吗?

A: 当前版本仅支持图像。视频水印需要专门的时序处理(留作未来扩展)。

### Q: 可以用 GPU 加速吗?

A: 当前是 CPU-only。但 L3a TrustMark 在有 GPU 时可以显著加速(模型已支持 ONNX)。

### Q: 被偷后能完全证明是我的吗?

A: 取决于攻击方式:
- 直接复制/裁切/社媒压缩: L1+L3+L4 全部命中 → 强证据
- AI 重生成: L3 失效, L4 仍有效(部分)
- 拍照重拍: L3 失效, L1+L4 仍有效
- 手动抄写: 攻击者已重新创作, 无法技术证明

### Q: 哪些平台会剥离水印?

A:
- ✅ 不剥离: 个人博客、GitHub、本地存储
- ⚠️ 部分剥离: 微博(压缩但保留元数据)
- ❌ 剥离 EXIF: 小红书、抖音、微信公众号
- L1-L4 不受元数据剥离影响

## 进阶用法

### 自定义签名检测阈值

```python
from mathmark import verify_image, load_config

cfg = load_config("~/.mathmark/config.yaml")
result = verify_image("stolen.png", cfg, threshold=0.6)
print(result.verdict, result.confidence)
```

### 编程式批处理

```python
from mathmark import WatermarkPipeline, WatermarkConfig
from mathmark.utils.image_io import load_image_rgb, save_image

cfg = WatermarkConfig(teacher_id="T001", teacher_name="王老师")
pipeline = WatermarkPipeline(cfg)

for i in range(10):
    img = load_image_rgb(f"lesson_{i}.png")
    result = pipeline.process(img, output_path=f"marked/lesson_{i}.png")
    print(f"lesson_{i}: {result.total_duration_ms:.0f}ms")
```

### 添加自定义签名维度

编辑 `signatures/<your_id>.json`,添加字段。匹配器会自动跳过未知字段。

### 自定义攻击模拟

```python
from mathmark.attacks import wechat_compress
from mathmark import verify_image

img = load_image_rgb("marked.png")
attacked = wechat_compress(img, quality=70)
save_image(attacked, "after_attack.png")

# 验证攻击后仍能识别
result = verify_image("after_attack.png", cfg)
print(result.verdict)
```

## 命令速查

| 命令 | 用途 |
|---|---|
| `mathmark init` | 初始化教师身份 |
| `mathmark sign` | 添加/编辑签名元素 |
| `mathmark embed` | 嵌入水印到图像 |
| `mathmark verify` | 验证图像归属 |
| `mathmark extract` | 提取水印 bits |
| `mathmark benchmark` | 性能测试 |
| `mathmark doctor` | 环境检查 |
| `mathmark --version` | 版本 |

## 文档导航

- [架构说明](architecture.md) - 6 层防御栈设计
- [攻击测试结果](attack_resistance.md) - 各种攻击下的表现
- [诚实局限](limitations.md) - 已知无法防御的攻击
