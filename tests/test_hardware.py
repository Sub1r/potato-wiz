"""
test_hardware.py
Tests for hardware detection (always safe — never crashes).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.hardware_detector import detect_hardware
from models.models import HardwareSpecs


def test_detect_hardware_returns_specs():
    """Hardware detection must never crash."""
    specs = detect_hardware()
    assert isinstance(specs, HardwareSpecs)


def test_hardware_has_sane_defaults():
    specs = detect_hardware()
    assert specs.ram_gb >= 1
    assert specs.vram_gb >= 1
    assert specs.cpu_cores >= 1
    assert specs.cpu_freq_ghz > 0
    assert specs.cpu
    assert specs.gpu
    assert specs.os_name
    assert 'x' in specs.resolution or specs.resolution


def test_hardware_to_dict():
    specs = detect_hardware()
    d = specs.to_dict()
    assert isinstance(d, dict)
    required_keys = ['cpu', 'gpu', 'ram_gb', 'vram_gb', 'os_name']
    for key in required_keys:
        assert key in d


def test_gpu_tier_returns_string():
    specs = HardwareSpecs(gpu='NVIDIA GeForce GTX 1650')
    assert isinstance(specs.gpu_tier, str)
    assert specs.gpu_tier in ('high-end', 'mid-high', 'mid', 'low-mid', 'low', 'unknown')


def test_gpu_tier_rtx_3080():
    specs = HardwareSpecs(gpu='NVIDIA GeForce RTX 3080')
    assert specs.gpu_tier in ('high-end', 'mid-high')


def test_gpu_tier_gtx_1060():
    specs = HardwareSpecs(gpu='NVIDIA GeForce GTX 1060 6GB')
    # GTX 1060 is classified as low (old budget card)
    assert specs.gpu_tier in ('mid', 'low-mid', 'low')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
