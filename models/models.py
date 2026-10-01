from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Benchmark matching result
# ---------------------------------------------------------------------------

# Valid match type literals
MATCH_TYPE_EXACT = "exact"          # GPU + CPU + resolution + preset all match
MATCH_TYPE_GPU = "gpu_match"        # GPU + resolution + preset match (CPU differs/absent)
MATCH_TYPE_APPROXIMATE = "approximate"  # GPU matches, some other field differs
MATCH_TYPE_NONE = "none"            # No usable benchmark found


@dataclass
class BenchmarkMatch:
    """
    Result returned by benchmark_service.find_benchmark().

    When ``found`` is False the FPS fields are None and the caller
    should fall back to the mathematical estimator.
    """
    found: bool = False
    match_type: str = MATCH_TYPE_NONE   # one of the MATCH_TYPE_* constants
    avg_fps: Optional[int] = None
    one_percent_low: Optional[int] = None
    source: str = ""
    source_url: str = ""
    confidence: str = ""               # "high" | "medium" | "low" | "fixture"
    record: Optional[Dict[str, Any]] = None  # the raw benchmark record
    reason: str = ""                   # human-readable explanation of the match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "match_type": self.match_type,
            "avg_fps": self.avg_fps,
            "one_percent_low": self.one_percent_low,
            "source": self.source,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class HardwareSpecs:
    cpu: str = "Unknown CPU"
    gpu: str = "Unknown GPU"
    ram_gb: int = 8
    vram_gb: int = 4
    os_name: str = "Windows"
    resolution: str = "1920x1080"
    refresh_rate: int = 60
    storage_type: str = "SSD"
    cpu_cores: int = 4
    cpu_freq_ghz: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu,
            "gpu": self.gpu,
            "ram_gb": self.ram_gb,
            "vram_gb": self.vram_gb,
            "os_name": self.os_name,
            "resolution": self.resolution,
            "refresh_rate": self.refresh_rate,
            "storage_type": self.storage_type,
            "cpu_cores": self.cpu_cores,
            "cpu_freq_ghz": self.cpu_freq_ghz,
        }

    @property
    def gpu_tier(self) -> str:
        """Estimate GPU tier from name."""
        gpu_lower = self.gpu.lower()
        high_end = ['4090', '4080', '4070 ti', '3090', '3080', '6900', '6800', '7900', '7800']
        mid_high = ['4070', '3070', '3060 ti', '6700', '6750', '7700', 'rx 6700']
        mid = ['3060', '4060', '2070', '2080', '6600', 'rx 6600', 'rx 5700']
        low_mid = ['1660', '2060', '3050', '4050', 'rx 5600', 'rx 5500']
        low = ['1650', '1060', '1050', 'rx 580', 'rx 570', 'rx 560', '750']

        for card in high_end:
            if card in gpu_lower:
                return 'high-end'
        for card in mid_high:
            if card in gpu_lower:
                return 'mid-high'
        for card in mid:
            if card in gpu_lower:
                return 'mid'
        for card in low_mid:
            if card in gpu_lower:
                return 'low-mid'
        for card in low:
            if card in gpu_lower:
                return 'low'
        return 'unknown'


@dataclass
class OptimizationResult:
    preset: str = "Medium"
    resolution: str = "1920x1080"
    upscaling: str = "Off"
    upscaling_mode: str = "Off"
    view_distance: str = "Medium"
    shadows: str = "Medium"
    effects: str = "Medium"
    textures: str = "Medium"
    anti_aliasing: str = "TAA"
    motion_blur: str = "Off"
    vsync: str = "Off"
    fps_limit: int = 60
    estimated_fps: str = "45-55"
    status: str = "Playable"
    notes: str = ""
    settings_list: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "resolution": self.resolution,
            "upscaling": self.upscaling,
            "upscaling_mode": self.upscaling_mode,
            "view_distance": self.view_distance,
            "shadows": self.shadows,
            "effects": self.effects,
            "textures": self.textures,
            "anti_aliasing": self.anti_aliasing,
            "motion_blur": self.motion_blur,
            "vsync": self.vsync,
            "fps_limit": self.fps_limit,
            "estimated_fps": self.estimated_fps,
            "status": self.status,
            "notes": self.notes,
            "settings_list": self.settings_list,
        }
