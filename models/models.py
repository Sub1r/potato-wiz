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


# ---------------------------------------------------------------------------
# AI Optimizer evidence and result models
# ---------------------------------------------------------------------------

# Evidence type constants — these MUST be used consistently so the UI can
# display the right label and the caller can distinguish evidence tiers.
EVIDENCE_MEASURED_BENCHMARK = "MEASURED_BENCHMARK"
EVIDENCE_PUBLISHED_BENCHMARK = "PUBLISHED_BENCHMARK"
EVIDENCE_COMMUNITY_REPORT = "COMMUNITY_REPORT"
EVIDENCE_OFFICIAL_INFO = "OFFICIAL_INFORMATION"
EVIDENCE_GUIDE = "GUIDE"
EVIDENCE_AI_INFERENCE = "AI_INFERENCE"
EVIDENCE_FALLBACK_ESTIMATE = "FALLBACK_ESTIMATE"

# AI provider identifiers
AI_PROVIDER_OPENROUTER = "openrouter"
AI_PROVIDER_OLLAMA = "ollama"
AI_PROVIDER_NONE = "none"           # deterministic fallback only

# AI result status
AI_STATUS_OK = "ok"
AI_STATUS_FALLBACK = "fallback"     # deterministic optimizer used
AI_STATUS_ERROR = "error"


@dataclass
class EvidenceItem:
    """
    A single piece of evidence collected during AI research.

    The ``type`` field must be one of the EVIDENCE_* constants above so the
    UI and callers can reliably distinguish measured data from inference.
    """
    title: str = ""
    url: str = ""
    domain: str = ""
    evidence_type: str = EVIDENCE_AI_INFERENCE
    claim: str = ""                  # specific factual claim drawn from this source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "type": self.evidence_type,
            "claim": self.claim,
        }


@dataclass
class AIRecommendedSettings:
    """
    Structured settings block returned by the AI.

    All fields have sensible defaults so partial AI output does not crash
    the validator.  The validator will fill gaps from the deterministic result.
    """
    graphics_preset: str = ""
    resolution: str = ""
    upscaling: str = ""
    view_distance: str = ""
    shadows: str = ""
    effects: str = ""
    textures: str = ""
    anti_aliasing: str = ""
    motion_blur: str = "Off"
    vsync: str = "Off"
    fps_limit: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graphics_preset": self.graphics_preset,
            "resolution": self.resolution,
            "upscaling": self.upscaling,
            "view_distance": self.view_distance,
            "shadows": self.shadows,
            "effects": self.effects,
            "textures": self.textures,
            "anti_aliasing": self.anti_aliasing,
            "motion_blur": self.motion_blur,
            "vsync": self.vsync,
            "fps_limit": self.fps_limit,
        }


@dataclass
class AIOptimizationResult:
    """
    Full structured result from the AI optimizer.

    ``status`` is one of AI_STATUS_* constants.
    ``fps_source`` documents where the FPS number came from — this is critical
    to ensure the AI never presents an estimate as a measured value.
    """
    # Core
    status: str = AI_STATUS_OK
    provider: str = AI_PROVIDER_NONE
    model: str = ""

    # Content
    summary: str = ""
    recommended_settings: Optional['AIRecommendedSettings'] = None
    changes: List[Dict[str, str]] = field(default_factory=list)

    # FPS — source MUST be documented
    estimated_fps: str = ""
    fps_source: str = EVIDENCE_FALLBACK_ESTIMATE   # one of EVIDENCE_* constants

    # Metadata
    confidence: str = "low"
    reasoning: str = ""
    evidence: List['EvidenceItem'] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Fallback deterministic result (always populated for safety)
    fallback_result: Optional['OptimizationResult'] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "summary": self.summary,
            "recommended_settings": (
                self.recommended_settings.to_dict()
                if self.recommended_settings else {}
            ),
            "changes": self.changes,
            "estimated_fps": self.estimated_fps,
            "fps_source": self.fps_source,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": [e.to_dict() for e in self.evidence],
            "warnings": self.warnings,
            "fallback": self.fallback_result.to_dict() if self.fallback_result else {},
        }


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
