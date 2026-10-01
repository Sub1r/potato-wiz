"""
hardware_detector.py
Safely detect system hardware using psutil and platform APIs.
Never crashes — all errors fall back to safe defaults.
"""

import platform
import sys
import subprocess
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from models.models import HardwareSpecs


def _safe(fn, default=None):
    """Execute fn() and return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


def _get_cpu_name() -> str:
    """Get CPU name from platform or subprocess."""
    # Try psutil first
    if PSUTIL_AVAILABLE:
        try:
            import cpuinfo  # optional
            info = cpuinfo.get_cpu_info()
            if info.get('brand_raw'):
                return info['brand_raw']
        except Exception:
            pass

    # Windows: use WMI via PowerShell
    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 'Get-WmiObject Win32_Processor | Select-Object -ExpandProperty Name'],
                capture_output=True, text=True, timeout=5
            )
            cpu = result.stdout.strip()
            if cpu:
                return cpu
        except Exception:
            pass

    # Linux: read /proc/cpuinfo
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except Exception:
            pass

    return platform.processor() or "Unknown CPU"


def _get_gpu_info() -> tuple:
    """Return (gpu_name, vram_gb). Safe on all platforms."""
    gpu_name = "Unknown GPU"
    vram_gb = 4  # safe default

    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 'Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json'],
                capture_output=True, text=True, timeout=8
            )
            if result.returncode == 0 and result.stdout.strip():
                import json
                raw = result.stdout.strip()
                data = json.loads(raw)
                # Handle both single object and list
                if isinstance(data, dict):
                    data = [data]
                if data:
                    gpu = data[0]
                    gpu_name = gpu.get('Name', 'Unknown GPU') or 'Unknown GPU'
                    adapter_ram = gpu.get('AdapterRAM')
                    if adapter_ram and adapter_ram > 0:
                        vram_gb = max(1, round(int(adapter_ram) / (1024 ** 3)))
        except Exception:
            pass

    elif sys.platform.startswith('linux'):
        try:
            result = subprocess.run(
                ['lspci'], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'VGA' in line or '3D' in line:
                    gpu_name = line.split(':')[-1].strip()
                    break
        except Exception:
            pass

    return gpu_name, vram_gb


def _get_ram_gb() -> int:
    """Get total system RAM in GB."""
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            return max(1, round(mem.total / (1024 ** 3)))
        except Exception:
            pass
    return 8  # safe default


def _get_cpu_details() -> tuple:
    """Return (cpu_cores, cpu_freq_ghz)."""
    cores = 4
    freq = 3.0

    if PSUTIL_AVAILABLE:
        try:
            cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
        except Exception:
            pass

        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                mhz = freq_info.max or freq_info.current or 3000
                freq = round(mhz / 1000, 1)
        except Exception:
            pass

    return cores, freq


def _get_os_name() -> str:
    """Get OS name."""
    try:
        sys_platform = platform.system()
        if sys_platform == 'Windows':
            ver = platform.version()
            release = platform.release()
            # Detect Win11
            try:
                build = int(ver.split('.')[2]) if '.' in ver else 0
                if build >= 22000:
                    return "Windows 11"
            except Exception:
                pass
            return f"Windows {release}"
        elif sys_platform == 'Darwin':
            return f"macOS {platform.mac_ver()[0]}"
        elif sys_platform == 'Linux':
            try:
                with open('/etc/os-release') as f:
                    for line in f:
                        if line.startswith('PRETTY_NAME='):
                            return line.split('=')[1].strip().strip('"')
            except Exception:
                pass
            return "Linux"
    except Exception:
        pass
    return "Unknown OS"


def _get_storage_type() -> str:
    """Detect if primary storage is SSD or HDD."""
    if not PSUTIL_AVAILABLE:
        return "SSD"
    try:
        partitions = psutil.disk_partitions()
        if sys.platform == 'win32':
            result = subprocess.run(
                ['powershell', '-Command',
                 'Get-PhysicalDisk | Select-Object MediaType | ConvertTo-Json'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and 'SSD' in result.stdout:
                return "SSD"
            elif result.returncode == 0 and 'HDD' in result.stdout:
                return "HDD"
    except Exception:
        pass
    return "SSD"


def _get_display_info() -> tuple:
    """Return (resolution_str, refresh_rate). Windows only via PowerShell."""
    resolution = "1920x1080"
    refresh_rate = 60

    if sys.platform == 'win32':
        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 'Add-Type -AssemblyName System.Windows.Forms; '
                 '[System.Windows.Forms.Screen]::PrimaryScreen | '
                 'Select-Object Bounds, @{N="Freq";E={' +
                 '[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width' +
                 '}} | ConvertTo-Json'],
                capture_output=True, text=True, timeout=5
            )
        except Exception:
            pass

        try:
            result = subprocess.run(
                ['powershell', '-Command',
                 '$s = Add-Type -MemberDefinition \'[DllImport("user32.dll")] public static extern int GetSystemMetrics(int nIndex);\' -Name Disp -Namespace Win32 -PassThru; '
                 '$w = $s::GetSystemMetrics(0); $h = $s::GetSystemMetrics(1); Write-Output "$w x $h"'],
                capture_output=True, text=True, timeout=5
            )
            out = result.stdout.strip()
            if 'x' in out:
                parts = out.replace(' ', '').split('x')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    resolution = f"{parts[0]}x{parts[1]}"
        except Exception:
            pass

    return resolution, refresh_rate


def detect_hardware() -> HardwareSpecs:
    """
    Detect hardware specs safely.
    Returns HardwareSpecs with real data where available,
    falling back to sensible defaults on any error.
    """
    cpu_name = _safe(_get_cpu_name, "Unknown CPU")
    gpu_name, vram_gb = _safe(_get_gpu_info, ("Unknown GPU", 4)) or ("Unknown GPU", 4)
    ram_gb = _safe(_get_ram_gb, 8)
    cpu_cores, cpu_freq = _safe(_get_cpu_details, (4, 3.0)) or (4, 3.0)
    os_name = _safe(_get_os_name, "Windows")
    storage_type = _safe(_get_storage_type, "SSD")
    resolution, refresh_rate = _safe(_get_display_info, ("1920x1080", 60)) or ("1920x1080", 60)

    return HardwareSpecs(
        cpu=cpu_name or "Unknown CPU",
        gpu=gpu_name or "Unknown GPU",
        ram_gb=int(ram_gb) if ram_gb else 8,
        vram_gb=int(vram_gb) if vram_gb else 4,
        os_name=os_name or "Windows",
        resolution=resolution or "1920x1080",
        refresh_rate=int(refresh_rate) if refresh_rate else 60,
        storage_type=storage_type or "SSD",
        cpu_cores=int(cpu_cores) if cpu_cores else 4,
        cpu_freq_ghz=float(cpu_freq) if cpu_freq else 3.0,
    )
