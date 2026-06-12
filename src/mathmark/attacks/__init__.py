"""Attacks module - 攻击模拟器(用于测试水印鲁棒性)"""
from . import diffusion_regen, local_blur, social_compress
from .diffusion_regen import (
    caption_guided_regen_simulation,
    full_diffusion_attack_simulation,
    noise_regen_simulation,
    semantic_regen_simulation,
)
from .local_blur import (
    add_noise,
    brightness_attack,
    contrast_attack,
    gaussian_blur_attack,
    gradcam_heuristic_mask,
    inpainting_attack,
    local_blur_attack,
)
from .social_compress import (
    crop_attack,
    douyin_compress,
    jpeg_recompress,
    resize_attack,
    wechat_compress,
    weibo_compress,
    xiaohongshu_compress,
)

__all__ = [
    "add_noise",
    "brightness_attack",
    "caption_guided_regen_simulation",
    "contrast_attack",
    "crop_attack",
    "diffusion_regen",
    "douyin_compress",
    "full_diffusion_attack_simulation",
    "gaussian_blur_attack",
    "gradcam_heuristic_mask",
    "inpainting_attack",
    "jpeg_recompress",
    "local_blur",
    "local_blur_attack",
    "noise_regen_simulation",
    "resize_attack",
    "semantic_regen_simulation",
    "social_compress",
    "wechat_compress",
    "weibo_compress",
    "xiaohongshu_compress",
]
