import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum

class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

@dataclass
class HardwareSnapshot:
    """Point-in-time hardware reading."""
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_pct: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_pct: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_available: bool = False
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        if self.gpu_available:
            return (
                f"VRAM {self.vram_used_mb:.0f}/{self.vram_total_mb:.0f} MB "
                f"({self.vram_pct:.1f}%) | "
                f"RAM {self.ram_pct:.1f}% | "
                f"GPU {self.gpu_temp_c:.0f}°C"
            )
        return (
            f"[CPU-only] RAM {self.ram_used_mb:.0f}/{self.ram_total_mb:.0f} MB "
            f"({self.ram_pct:.1f}%)"
        )

@dataclass
class P2PPacket:
    """Mock P2P weight-exchange packet."""
    node_id: str
    precision: Precision
    arch: str                  # "arm" | "x86"
    layer_names: list
    payload_bytes: bytes
    metadata: dict = field(default_factory=dict)

    def header(self) -> bytes:
        bits_map = {Precision.FP32: 32, Precision.FP16: 16,
                    Precision.INT8: 8,  Precision.INT4: 4}
        arch_id  = 1 if self.arch == "arm" else 0
        node_bytes = self.node_id.encode()[:16].ljust(16, b"\x00")

        return (
            b"MGOP"
            + struct.pack("<I", 1)
            + struct.pack("<I", bits_map[self.precision])
            + struct.pack("<I", arch_id)
            + node_bytes
        )

    def serialize(self) -> bytes:
        meta_bytes = json.dumps(self.metadata).encode()
        meta_len   = struct.pack("<I", len(meta_bytes))
        return self.header() + meta_len + meta_bytes + self.payload_bytes

    def __repr__(self) -> str:
        return (
            f"<P2PPacket node={self.node_id} precision={self.precision.value} "
            f"arch={self.arch} payload={len(self.payload_bytes)} B>"
        )
