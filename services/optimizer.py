"""
optimizer.py
Generates recommended graphics settings based on hardware and user preferences.
"""

from typing import Dict, Any, List
from models.models import HardwareSpecs, OptimizationResult
from services.fps_estimator import estimate_fps_with_metadata, fps_to_status, fps_to_display


# Preset definitions per tier
PRESETS_BY_TIER = {
    'high-end': {
        'fps':     {'preset': 'Ultra',  'resolution': '2560x1440', 'upscaling': 'Off',      'view_distance': 'Ultra',  'shadows': 'Ultra',  'effects': 'Ultra',  'textures': 'Ultra',  'aa': 'TAA', 'fps_limit': 0},
        'balanced':{'preset': 'Ultra',  'resolution': '2560x1440', 'upscaling': 'Off',      'view_distance': 'Ultra',  'shadows': 'High',   'effects': 'Ultra',  'textures': 'Ultra',  'aa': 'TAA', 'fps_limit': 0},
        'quality': {'preset': 'Ultra',  'resolution': '3840x2160', 'upscaling': 'DLSS Quality','view_distance': 'Ultra','shadows': 'Ultra', 'effects': 'Ultra',  'textures': 'Ultra',  'aa': 'TAA', 'fps_limit': 60},
    },
    'mid-high': {
        'fps':     {'preset': 'High',   'resolution': '1920x1080', 'upscaling': 'Off',      'view_distance': 'High',   'shadows': 'Medium', 'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 0},
        'balanced':{'preset': 'High',   'resolution': '1920x1080', 'upscaling': 'Off',      'view_distance': 'High',   'shadows': 'High',   'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 0},
        'quality': {'preset': 'High',   'resolution': '2560x1440', 'upscaling': 'FSR Quality','view_distance': 'High', 'shadows': 'High',   'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
    },
    'mid': {
        'fps':     {'preset': 'Medium', 'resolution': '1920x1080', 'upscaling': 'Balanced', 'view_distance': 'Medium', 'shadows': 'Low',    'effects': 'Medium', 'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
        'balanced':{'preset': 'High',   'resolution': '1920x1080', 'upscaling': 'Balanced', 'view_distance': 'High',   'shadows': 'Medium', 'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
        'quality': {'preset': 'High',   'resolution': '1920x1080', 'upscaling': 'Quality',  'view_distance': 'High',   'shadows': 'Medium', 'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
    },
    'low-mid': {
        'fps':     {'preset': 'Low',    'resolution': '1920x1080', 'upscaling': 'Quality',  'view_distance': 'Low',    'shadows': 'Low',    'effects': 'Low',    'textures': 'Medium', 'aa': 'Off', 'fps_limit': 60},
        'balanced':{'preset': 'Medium', 'resolution': '1920x1080', 'upscaling': 'Quality',  'view_distance': 'Medium', 'shadows': 'Low',    'effects': 'Medium', 'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
        'quality': {'preset': 'Medium', 'resolution': '1920x1080', 'upscaling': 'Balanced', 'view_distance': 'Medium', 'shadows': 'Medium', 'effects': 'Medium', 'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
    },
    'low': {
        'fps':     {'preset': 'Low',    'resolution': '1280x720',  'upscaling': 'Quality',  'view_distance': 'Low',    'shadows': 'Low',    'effects': 'Low',    'textures': 'Low',    'aa': 'Off', 'fps_limit': 30},
        'balanced':{'preset': 'Low',    'resolution': '1280x720',  'upscaling': 'Quality',  'view_distance': 'Low',    'shadows': 'Low',    'effects': 'Low',    'textures': 'Medium', 'aa': 'Off', 'fps_limit': 30},
        'quality': {'preset': 'Medium', 'resolution': '1280x720',  'upscaling': 'Quality',  'view_distance': 'Low',    'shadows': 'Low',    'effects': 'Medium', 'textures': 'Medium', 'aa': 'TAA', 'fps_limit': 30},
    },
    'unknown': {
        'fps':     {'preset': 'Medium', 'resolution': '1920x1080', 'upscaling': 'Balanced', 'view_distance': 'Medium', 'shadows': 'Low',    'effects': 'Medium', 'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
        'balanced':{'preset': 'High',   'resolution': '1920x1080', 'upscaling': 'Balanced', 'view_distance': 'High',   'shadows': 'Medium', 'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
        'quality': {'preset': 'High',   'resolution': '1920x1080', 'upscaling': 'Quality',  'view_distance': 'High',   'shadows': 'High',   'effects': 'High',   'textures': 'High',   'aa': 'TAA', 'fps_limit': 60},
    },
}

SETTING_EXPLANATIONS = {
    'view_distance': {
        'Ultra':  'Maximum draw distance. Best for open-world exploration. High GPU cost.',
        'High':   'Far draw distance with good performance balance.',
        'Medium': 'Moderate draw distance. Reduces GPU load in open areas.',
        'Low':    'Short draw distance. Significantly improves GPU performance.',
    },
    'shadows': {
        'Ultra':  'Maximum shadow resolution. Very high GPU cost. For top-end hardware only.',
        'High':   'Detailed shadows with good balance of quality and performance.',
        'Medium': 'Good shadow quality with moderate GPU cost. Recommended sweet spot.',
        'Low':    'Basic shadows to save GPU performance.',
        'Off':    'Disables dynamic shadows. Large performance gain but visual loss.',
    },
    'effects': {
        'Ultra':  'Maximum particle and post-process effects. Very demanding.',
        'High':   'Excellent effects quality with acceptable performance cost.',
        'Medium': 'Good effects at moderate GPU cost.',
        'Low':    'Minimal effects to save GPU load.',
    },
    'textures': {
        'Ultra':  'Maximum texture resolution. Requires 8GB+ VRAM.',
        'High':   'Detailed textures. Works well on 4GB–6GB VRAM.',
        'Medium': 'Moderate texture quality. Works well on 2–4GB VRAM.',
        'Low':    'Low-resolution textures. Best for limited VRAM.',
    },
    'upscaling': {
        'Off':                'No upscaling. Renders at native resolution.',
        'Quality':            'High quality upscaling. Near-native sharpness with ~25% FPS boost.',
        'Balanced':           'Balanced upscaling. Good sharpness with ~45% FPS boost.',
        'Performance':        'Performance upscaling. Some softness but large FPS gain.',
        'Ultra Performance':  'Maximum FPS boost. Noticeable quality reduction.',
        'DLSS Quality':       'NVIDIA DLSS Quality mode. Near-native sharpness.',
        'DLSS Balanced':      'NVIDIA DLSS Balanced. Good sharpness with significant FPS boost.',
        'FSR Quality':        'AMD FSR Quality mode. Near-native sharpness with ~25% FPS boost.',
        'FSR Balanced':       'AMD FSR Balanced mode. Good sharpness with ~45% FPS boost.',
        'FSR Ultra Perf':     'AMD FSR Ultra Performance. Maximum FPS gain.',
    },
}


def _build_settings_list(settings: Dict, game_settings_detail: list) -> List[Dict]:
    """
    Build a settings list whose values always come from the *computed* settings
    dict, never from the static game-level settings_detail.

    If the game provides a settings_detail list, we use its ``why``
    explanations for any setting that matches by name — but we never override
    the *value* that was actually selected by the optimizer.

    This prevents the UI from describing settings that differ from what the
    optimizer actually chose.
    """
    # Build a lookup from setting name → why explanation from the game data
    game_why: Dict[str, str] = {}
    for entry in (game_settings_detail or []):
        if isinstance(entry, dict) and "name" in entry and "why" in entry:
            game_why[entry["name"].strip().lower()] = entry["why"]

    def _why(setting_name: str, fallback: str) -> str:
        return game_why.get(setting_name.strip().lower(), fallback)

    return [
        {
            "name": "Graphics Preset",
            "value": settings["preset"],
            "why": _why("graphics preset", "Recommended preset for your hardware tier."),
        },
        {
            "name": "Resolution",
            "value": settings["resolution"].replace("x", "×"),
            "why": _why("resolution", "Optimal resolution for performance/quality balance."),
        },
        {
            "name": "FSR / Upscaling",
            "value": settings["upscaling"],
            "why": _why(
                "fsr / upscaling",
                SETTING_EXPLANATIONS["upscaling"].get(settings["upscaling"], "Upscaling setting."),
            ),
        },
        {
            "name": "View Distance",
            "value": settings["view_distance"],
            "why": _why(
                "view distance",
                SETTING_EXPLANATIONS["view_distance"].get(settings["view_distance"], ""),
            ),
        },
        {
            "name": "Shadows",
            "value": settings["shadows"],
            "why": _why(
                "shadows",
                SETTING_EXPLANATIONS["shadows"].get(settings["shadows"], ""),
            ),
        },
        {
            "name": "Textures",
            "value": settings["textures"],
            "why": _why(
                "textures",
                SETTING_EXPLANATIONS["textures"].get(settings["textures"], ""),
            ),
        },
        {
            "name": "Effects",
            "value": settings["effects"],
            "why": _why(
                "effects",
                SETTING_EXPLANATIONS["effects"].get(settings["effects"], ""),
            ),
        },
        {
            "name": "Anti-Aliasing",
            "value": settings["aa"],
            "why": _why("anti-aliasing", "Smooths jagged edges in the image."),
        },
        {
            "name": "Motion Blur",
            "value": "Off",
            "why": _why("motion blur", "Disabling motion blur improves clarity and perceived sharpness."),
        },
        {
            "name": "V-Sync",
            "value": "Off",
            "why": _why("v-sync", "Disabled V-Sync reduces input latency."),
        },
        {
            "name": "FPS Limit",
            "value": str(settings["fps_limit"]) if settings["fps_limit"] else "Unlimited",
            "why": _why("fps limit", "Caps FPS to stabilize frame pacing."),
        },
    ]


def generate_recommendations(
    hardware: HardwareSpecs,
    game: Dict,
    target_fps: int = 60,
    priority: str = 'balanced',
    resolution: str = None,
) -> OptimizationResult:
    """
    Generate optimized settings for a game given hardware specs.
    Priority: 'fps' | 'balanced' | 'quality'
    """
    tier = hardware.gpu_tier
    priority = priority.lower() if priority else 'balanced'

    # Get base settings from tier/priority matrix
    tier_presets = PRESETS_BY_TIER.get(tier, PRESETS_BY_TIER['unknown'])
    settings = dict(tier_presets.get(priority, tier_presets['balanced']))

    # Override resolution if provided by user
    if resolution and 'x' in resolution:
        settings['resolution'] = resolution

    # Apply target FPS adjustments
    if target_fps <= 30 and tier in ('mid', 'low-mid', 'low'):
        pass  # keep current settings
    elif target_fps >= 120 and tier in ('low', 'low-mid'):
        # Need to go lower for high FPS target
        settings['preset'] = 'Low'
        settings['upscaling'] = 'Performance'
        settings['shadows'] = 'Low'
        settings['effects'] = 'Low'

    # Estimate FPS — uses benchmark data when available, falls back to math
    fps_meta = estimate_fps_with_metadata(
        hardware,
        game.get('slug', ''),
        preset=settings['preset'],
        resolution=settings['resolution'],
        upscaling_mode=settings['upscaling'],
    )
    fps_low = fps_meta['fps_low']
    fps_high = fps_meta['fps_high']

    status = fps_to_status(fps_low, fps_high)
    fps_display = fps_to_display(fps_low, fps_high)

    # Determine upscaling type label
    upscaling_str = settings['upscaling']
    upscaling_mode = upscaling_str if upscaling_str != 'Off' else 'Off'

    # Performance notes
    avg_fps = (fps_low + fps_high) / 2
    if avg_fps >= 100:
        notes = "Your system handles this game exceptionally well. You can increase settings further."
    elif avg_fps >= 70:
        notes = "Excellent performance. Smooth gameplay expected across all game areas."
    elif avg_fps >= 55:
        notes = "Good performance. You should enjoy smooth gameplay at these settings."
    elif avg_fps >= 35:
        notes = "Playable performance. Consider lowering shadows and effects for more headroom."
    else:
        notes = "Performance is limited. Try lowering resolution or enabling upscaling for better results."

    # Get detailed settings list from game data if available
    game_settings_detail = game.get('settings_detail', [])

    result = OptimizationResult(
        preset=settings['preset'],
        resolution=settings['resolution'],
        upscaling='FSR 2.0' if upscaling_mode not in ('Off', 'DLSS Quality', 'DLSS Balanced') else ('DLSS 3.0' if 'DLSS' in upscaling_mode else 'Off'),
        upscaling_mode=upscaling_mode,
        view_distance=settings['view_distance'],
        shadows=settings['shadows'],
        effects=settings['effects'],
        textures=settings['textures'],
        anti_aliasing=settings['aa'],
        motion_blur='Off',
        vsync='Off',
        fps_limit=settings['fps_limit'] if settings['fps_limit'] else target_fps,
        estimated_fps=fps_display,
        status=status,
        notes=notes,
        settings_list=_build_settings_list(settings, game_settings_detail),
    )

    return result
