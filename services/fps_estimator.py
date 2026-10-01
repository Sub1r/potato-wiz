"""
fps_estimator.py
Estimates expected FPS for a given game and hardware configuration.
"""

from typing import Dict, Any, Tuple
from models.models import HardwareSpecs


# GPU performance multipliers relative to GTX 1080 baseline (1.0)
GPU_PERFORMANCE_MAP = {
    # NVIDIA RTX 40 series
    'rtx 4090': 3.5, 'rtx 4080': 2.8, 'rtx 4070 ti': 2.3,
    'rtx 4070': 2.0, 'rtx 4060 ti': 1.6, 'rtx 4060': 1.3,
    # NVIDIA RTX 30 series
    'rtx 3090': 2.9, 'rtx 3080 ti': 2.7, 'rtx 3080': 2.4,
    'rtx 3070 ti': 2.0, 'rtx 3070': 1.8, 'rtx 3060 ti': 1.6,
    'rtx 3060': 1.3, 'rtx 3050': 1.0,
    # NVIDIA RTX 20 series
    'rtx 2080 ti': 2.1, 'rtx 2080': 1.8, 'rtx 2070': 1.5,
    'rtx 2060': 1.3, 'rtx 2060 super': 1.4,
    # NVIDIA GTX 16/10 series
    'gtx 1660 ti': 1.2, 'gtx 1660': 1.1, 'gtx 1650': 0.8,
    'gtx 1080 ti': 1.7, 'gtx 1080': 1.0, 'gtx 1070 ti': 0.9,
    'gtx 1070': 0.85, 'gtx 1060 6gb': 0.75, 'gtx 1060': 0.7,
    'gtx 1050 ti': 0.55, 'gtx 1050': 0.45,
    # AMD RX 7000 series
    'rx 7900 xtx': 3.0, 'rx 7900 xt': 2.6, 'rx 7800 xt': 2.0,
    'rx 7700 xt': 1.7, 'rx 7600': 1.4,
    # AMD RX 6000 series
    'rx 6900 xt': 2.6, 'rx 6800 xt': 2.3, 'rx 6800': 2.0,
    'rx 6750 xt': 1.8, 'rx 6700 xt': 1.6, 'rx 6700': 1.45,
    'rx 6650 xt': 1.35, 'rx 6600 xt': 1.25, 'rx 6600': 1.15,
    'rx 6500 xt': 0.7,
    # AMD RX 5000 series
    'rx 5700 xt': 1.4, 'rx 5700': 1.2, 'rx 5600 xt': 1.05,
    'rx 5500 xt': 0.75,
    # AMD older
    'rx 580': 0.65, 'rx 570': 0.55, 'rx 480': 0.6,
}

# Preset performance cost (relative FPS hit vs Low preset)
PRESET_COSTS = {
    'Low': 1.0,
    'Normal': 0.88,
    'Medium': 0.78,
    'High': 0.65,
    'Very High': 0.55,
    'Ultra': 0.45,
    'Maximum': 0.42,
    'Extreme': 0.38,
    'Ultra+RT': 0.30,
}

# Resolution scaling factors (relative to 1080p = 1.0)
RESOLUTION_COSTS = {
    '1280x720': 0.56,
    '1600x900': 0.75,
    '1920x1080': 1.0,
    '2560x1440': 1.78,
    '3840x2160': 4.0,
}

# Upscaling FPS multipliers
UPSCALING_BOOST = {
    'Off': 1.0,
    'Quality': 1.25,
    'Balanced': 1.45,
    'Performance': 1.65,
    'Ultra Performance': 1.85,
    'DLSS Quality': 1.3,
    'DLSS Balanced': 1.5,
    'FSR Quality': 1.25,
    'FSR Balanced': 1.45,
    'FSR Ultra Perf': 1.85,
    'XeSS Quality': 1.25,
}

# Base FPS on GTX 1080 @ 1080p / High preset
GAME_BASE_FPS = {
    'palworld': 55,
    'elden-ring': 58,
    'cyberpunk-2077': 40,
    'baldurs-gate-3': 60,
    'hogwarts-legacy': 45,
    'red-dead-redemption-2': 55,
    'gta-v': 80,
    'forza-horizon-5': 70,
    'the-witcher-3': 75,
    'starfield': 48,
}


def _get_gpu_multiplier(gpu_name: str) -> float:
    """Look up GPU multiplier from name."""
    gpu_lower = gpu_name.lower()
    best_match = 0.7  # default: roughly GTX 1060
    best_len = 0
    for key, mult in GPU_PERFORMANCE_MAP.items():
        if key in gpu_lower and len(key) > best_len:
            best_match = mult
            best_len = len(key)
    return best_match


def estimate_fps(
    hardware: HardwareSpecs,
    game_slug: str,
    preset: str = 'High',
    resolution: str = '1920x1080',
    upscaling_mode: str = 'Off',
) -> Tuple[int, int]:
    """
    Returns (fps_low, fps_high) estimated range.
    """
    base_fps = GAME_BASE_FPS.get(game_slug, 55)
    gpu_mult = _get_gpu_multiplier(hardware.gpu)
    preset_cost = PRESET_COSTS.get(preset, 0.65)
    res_cost = RESOLUTION_COSTS.get(resolution, 1.0)
    upscale_boost = UPSCALING_BOOST.get(upscaling_mode, 1.0)

    # RAM penalty below 12GB
    ram_mult = 1.0
    if hardware.ram_gb < 8:
        ram_mult = 0.8
    elif hardware.ram_gb < 12:
        ram_mult = 0.92

    estimated = base_fps * gpu_mult * preset_cost / res_cost * upscale_boost * ram_mult

    fps_low = max(5, int(estimated * 0.88))
    fps_high = max(fps_low + 5, int(estimated * 1.12))

    return fps_low, fps_high


def fps_to_status(fps_low: int, fps_high: int) -> str:
    avg = (fps_low + fps_high) / 2
    if avg >= 100:
        return "Excellent"
    elif avg >= 75:
        return "Very Good"
    elif avg >= 55:
        return "Playable"
    elif avg >= 35:
        return "Low"
    else:
        return "Unplayable"


def fps_to_display(fps_low: int, fps_high: int) -> str:
    return f"{fps_low}–{fps_high}"
