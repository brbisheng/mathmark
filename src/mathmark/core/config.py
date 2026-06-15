"""配置加载

支持 YAML 和 JSON 两种格式,统一入口加载 WatermarkConfig。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import yaml

from .types import (
    C2PASettings,
    CoxSettings,
    DWTSettings,
    LayerType,
    MathSignature,
    MetadataSettings,
    SemanticSettings,
    TrustMarkSettings,
    VisibleSettings,
    WatermarkConfig,
)

PathLike = Union[str, Path]


def load_signature(path: PathLike) -> MathSignature:
    """从 JSON 文件加载数学签名"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return MathSignature.from_dict(data)


def save_signature(signature: MathSignature, path: PathLike) -> None:
    """保存数学签名到 JSON"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(signature.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_layers(layers: Any) -> set[LayerType]:
    """解析 layers 配置 (支持 'all', 列表, 或逗号分隔字符串).

    Audit B6: bare "L3" is ambiguous (matches L3a/L3b/L3c all three) and the
    old `startswith` branch returned the FIRST matching layer silently,
    which is wrong for an unknown specifier. Now we match either the full
    value (e.g. "L3a_trustmark") OR the prefix-with-underscore (e.g. "L3a"),
    and reject bare "L1"/"L3" with a clear error.
    """
    if layers == "all":
        return set(LayerType)
    if isinstance(layers, str):
        layers = [s.strip() for s in layers.split(",")]
    if isinstance(layers, list):
        result = set()
        layer_map = {lt.value: lt for lt in LayerType}
        for item in layers:
            if not isinstance(item, str):
                continue
            if item in layer_map:
                result.add(layer_map[item])
                continue
            # Match by "L<n><letter>_" prefix (e.g. "L3a" -> "L3a_trustmark")
            prefix = item.split("_")[0]
            matches = [lt for lt in LayerType if lt.value == item or lt.value.split("_")[0] == prefix]
            if not matches:
                raise ValueError(
                    f"Unknown layer specifier: '{item}'. "
                    f"Use one of: {sorted(layer_map.keys())} or 'all'."
                )
            if len(matches) > 1:
                # Ambiguous like "L3" — accept the specifier only if it's
                # already underscore-separated (L3a, L3b, L3c), otherwise
                # reject with a clear error.
                if item != prefix:
                    result.update(matches)
                else:
                    raise ValueError(
                        f"Ambiguous layer specifier: '{item}' matches {sorted(m.value for m in matches)}. "
                        f"Use one of: {sorted(m.value for m in matches)}."
                    )
            else:
                result.add(matches[0])
        return result
    return set(LayerType)


def load_config(path: PathLike) -> WatermarkConfig:
    """从 YAML 或 JSON 文件加载完整配置"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        # 尝试自动检测
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = json.loads(text)

    return _build_config(data)


def _build_config(data: dict[str, Any]) -> WatermarkConfig:
    """从 dict 构造 WatermarkConfig"""
    teacher = data.get("teacher", {})
    watermark = data.get("watermark", {})
    verification = data.get("verification", {})

    # 解析 enabled_layers
    enabled = watermark.get("enabled_layers", "all")
    enabled_layers = _parse_layers(enabled)

    # 解析各层配置
    visible_d = watermark.get("visible", {})
    trustmark_d = watermark.get("trustmark", {})
    dwt_d = watermark.get("dwt_dct_svd", {})
    cox_d = watermark.get("cox_spread", {})
    semantic_d = watermark.get("semantic", {})
    metadata_d = watermark.get("metadata", {})
    c2pa_d = watermark.get("c2pa", {})

    # 加载签名配置(如果有)
    signature = None
    if "signature_config" in semantic_d:
        sig_path = Path(semantic_d["signature_config"]).expanduser()
        if sig_path.exists():
            signature = load_signature(sig_path)

    return WatermarkConfig(
        teacher_id=teacher.get("id", "unknown"),
        teacher_name=teacher.get("name", ""),
        teacher_public_key_path=Path(teacher["public_key_path"]).expanduser() if teacher.get("public_key_path") else None,
        teacher_private_key_path=Path(teacher["private_key_path"]).expanduser() if teacher.get("private_key_path") else None,
        enabled_layers=enabled_layers,
        visible=VisibleSettings(
            text=visible_d.get("text", "© {teacher_id} {teacher_name}"),
            position=visible_d.get("position", "tiled"),
            opacity=visible_d.get("opacity", 0.20),
            perturbation_strength=visible_d.get("perturbation_strength", 0.02),
            enable_perturbation=visible_d.get("enable_perturbation", True),
        ),
        trustmark=TrustMarkSettings(
            model_path=Path(trustmark_d["model_path"]).expanduser() if trustmark_d.get("model_path") else None,
            use_quantized=trustmark_d.get("use_quantized", True),
        ),
        dwt=DWTSettings(
            alpha=dwt_d.get("alpha", 0.2),
            block_size=dwt_d.get("block_size", 4),
        ),
        cox=CoxSettings(
            strength=cox_d.get("strength", 0.05),
            seed=cox_d.get("seed", 42),
        ),
        semantic=SemanticSettings(
            signature=signature,
            injection_strength=semantic_d.get("injection_strength", 0.5),
            auto_suggest=semantic_d.get("auto_suggest", True),
        ),
        metadata=MetadataSettings(
            write_exif=metadata_d.get("exif", True),
            write_xmp=metadata_d.get("xmp", True),
            copyright=metadata_d.get("copyright", ""),
            contact=metadata_d.get("contact", ""),
            custom_fields=metadata_d.get("custom_fields", {}),
        ),
        c2pa=C2PASettings(
            enable=c2pa_d.get("enable", True),
            algorithm=c2pa_d.get("algorithm", "ES256"),
        ),
        similarity_threshold=verification.get("similarity_threshold", 0.75),
        legal_export=verification.get("legal_export", True),
    )


def save_config(config: WatermarkConfig, path: PathLike) -> None:
    """保存配置到 YAML 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "teacher": {
            "id": config.teacher_id,
            "name": config.teacher_name,
            "public_key_path": str(config.teacher_public_key_path) if config.teacher_public_key_path else None,
            "private_key_path": str(config.teacher_private_key_path) if config.teacher_private_key_path else None,
        },
        "watermark": {
            "enabled_layers": sorted([lt.value for lt in config.enabled_layers]),
            "visible": {
                "text": config.visible.text,
                "position": config.visible.position,
                "opacity": config.visible.opacity,
                "perturbation_strength": config.visible.perturbation_strength,
                "enable_perturbation": config.visible.enable_perturbation,
            },
            "trustmark": {
                "model_path": str(config.trustmark.model_path) if config.trustmark.model_path else None,
                "use_quantized": config.trustmark.use_quantized,
            },
            "dwt_dct_svd": {
                "alpha": config.dwt.alpha,
                "block_size": config.dwt.block_size,
            },
            "cox_spread": {
                "strength": config.cox.strength,
                "seed": config.cox.seed,
            },
            "semantic": {
                "injection_strength": config.semantic.injection_strength,
                "auto_suggest": config.semantic.auto_suggest,
            },
            "metadata": {
                "exif": config.metadata.write_exif,
                "xmp": config.metadata.write_xmp,
                "copyright": config.metadata.copyright,
                "contact": config.metadata.contact,
                "custom_fields": config.metadata.custom_fields,
            },
            "c2pa": {
                "enable": config.c2pa.enable,
                "algorithm": config.c2pa.algorithm,
            },
        },
        "verification": {
            "similarity_threshold": config.similarity_threshold,
            "legal_export": config.legal_export,
        },
    }
    if path.suffix in (".yaml", ".yml"):
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_default_config(
    teacher_id: str,
    teacher_name: str = "",
    path: Optional[PathLike] = None,
) -> "WatermarkConfig":
    """创建默认配置 (返回 config 对象, 可选保存到 path)"""
    config = WatermarkConfig(teacher_id=teacher_id, teacher_name=teacher_name)
    if path is not None:
        save_config(config, path)
    return config
