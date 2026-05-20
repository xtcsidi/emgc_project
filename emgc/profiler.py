import logging
from typing import Optional
import psutil
from .datatypes import HardwareSnapshot

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
    logging.warning("pynvml not available – falling back to psutil (CPU/RPi mode).")

log = logging.getLogger("MemoryGatedOptimizer")

def get_nvml_handle(gpu_index: int) -> Optional[object]:
    if _NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            info = pynvml.nvmlDeviceGetName(handle)
            log.info(f"GPU detected: {info.decode() if isinstance(info, bytes) else info}")
            return handle
        except pynvml.NVMLError as e:
            log.warning(f"NVML handle error: {e}. Falling back to psutil.")
    return None

def profile_hardware(handle: Optional[object]) -> HardwareSnapshot:
    """Return a fresh HardwareSnapshot (GPU via pynvml, else psutil fallback)."""
    snap = HardwareSnapshot()

    mem = psutil.virtual_memory()
    snap.ram_total_mb = mem.total / 1024**2
    snap.ram_used_mb  = mem.used  / 1024**2
    snap.ram_pct      = mem.percent

    if _NVML_AVAILABLE and handle:
        try:
            vram       = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp       = pynvml.nvmlDeviceGetTemperature(
                             handle, pynvml.NVML_TEMPERATURE_GPU)
            snap.vram_total_mb = vram.total / 1024**2
            snap.vram_used_mb  = vram.used  / 1024**2
            snap.vram_pct      = vram.used   / vram.total
            snap.gpu_temp_c    = float(temp)
            snap.gpu_available = True
        except pynvml.NVMLError as e:
            log.warning(f"NVML read error: {e}")

    if not snap.gpu_available:
        try:
            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            for key in ("cpu_thermal", "coretemp", "k10temp"):
                if key in temps:
                    snap.gpu_temp_c = temps[key][0].current
                    break
        except AttributeError:
            pass

    return snap
