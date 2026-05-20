import io
import time
import logging
import torch
import torch.nn as nn
from typing import Optional

from .datatypes import Precision, HardwareSnapshot, P2PPacket
from .profiler import get_nvml_handle, profile_hardware
from .quantization import to_fp16, to_int8, to_int4, _BNB_AVAILABLE
from .pruning import apply_thermal_pruning, compute_sparsity

log = logging.getLogger("MemoryGatedOptimizer")

class MemoryGatedOptimizer:
    def __init__(
        self,
        model: nn.Module,
        node_id: str = "dfl-node-0",
        gpu_index: int = 0,
        vram_threshold: float = 0.80,
        temp_threshold: float = 80.0,
        prune_amount: float = 0.20,
        poll_interval: float = 5.0,
        target_arch: str = "x86",
    ):
        self.model          = model
        self.node_id        = node_id
        self.gpu_index      = gpu_index
        self.vram_threshold = vram_threshold
        self.temp_threshold = temp_threshold
        self.prune_amount   = prune_amount
        self.poll_interval  = poll_interval
        self.target_arch    = target_arch

        self.device = torch.device(f"cuda:{self.gpu_index}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.current_precision: Precision = Precision.FP16
        self._pruned_layers: set          = set()
        
        # Initialize hardware handles
        self._handle = get_nvml_handle(self.gpu_index)

        # Baseline: FP16 on GPU, FP32 on CPU
        if self.device.type == "cuda":
            self.model.half()
            self.current_precision = Precision.FP16
            log.info(f"Model initialized at FP16 on CUDA baseline ({self.device}).")
        else:
            self.current_precision = Precision.FP32
            log.info("Model initialized at FP32 on CPU baseline (VRAM gating disabled).")

    def profile_hardware(self) -> HardwareSnapshot:
        return profile_hardware(self._handle)

    def apply_elastic_quantization(self, snap: HardwareSnapshot):
        if not snap.gpu_available:
            log.debug("No GPU – skipping VRAM-based quantization gate.")
            return

        vram_pct = snap.vram_pct

        if vram_pct > self.vram_threshold:
            if self.current_precision == Precision.FP16:
                log.warning(f"VRAM {vram_pct:.1%} > {self.vram_threshold:.0%} → switching FP16→INT8")
                self.current_precision = to_int8(self.model)
            elif self.current_precision == Precision.INT8:
                log.warning(f"VRAM still high ({vram_pct:.1%}) → switching INT8→INT4")
                self.current_precision = to_int4(self.model)
            else:
                log.warning("Already at INT4 – cannot compress further.")
        elif vram_pct < self.vram_threshold * 0.70:
            if self.current_precision != Precision.FP16:
                log.info(f"VRAM eased ({vram_pct:.1%}) → restoring FP16")
                self.current_precision = to_fp16(self.model)

    def apply_thermal_pruning(self, snap: HardwareSnapshot):
        apply_thermal_pruning(
            model=self.model,
            snap=snap,
            temp_threshold=self.temp_threshold,
            prune_amount=self.prune_amount,
            pruned_layers=self._pruned_layers
        )

    def export_weights_p2p(
        self,
        arch: Optional[str] = None,
        layer_filter: Optional[list] = None,
    ) -> P2PPacket:
        target_arch = arch or self.target_arch
        buf = io.BytesIO()

        state = self.model.state_dict()
        selected_keys = [
            k for k in state
            if layer_filter is None or any(f in k for f in layer_filter)
        ]

        torch.save({k: state[k] for k in selected_keys}, buf)
        payload = buf.getvalue()

        metadata = {
            "precision":     self.current_precision.value,
            "layer_count":   len(selected_keys),
            "sparsity":      round(compute_sparsity(self.model), 4),
            "bnb_available": _BNB_AVAILABLE,
            "arch":          target_arch,
            "framework":     f"torch-{torch.__version__}",
            "timestamp":     time.time(),
        }

        packet = P2PPacket(
            node_id      = self.node_id,
            precision    = self.current_precision,
            arch         = target_arch,
            layer_names  = selected_keys,
            payload_bytes= payload,
            metadata     = metadata,
        )

        log.info(
            f"[P2P Export] {packet} | layers={len(selected_keys)} | "
            f"arch={target_arch} | sparsity={metadata['sparsity']:.2%}"
        )
        return packet

    def import_weights_p2p(self, packet: P2PPacket, strict: bool = False):
        buf = io.BytesIO(packet.payload_bytes)
        remote_state = torch.load(buf, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(remote_state, strict=strict)
        log.info(
            f"[P2P Import] from {packet.node_id} | precision={packet.precision.value} | "
            f"missing={missing} | unexpected={unexpected}"
        )

    def run(self, max_rounds: int = 10, callback=None):
        log.info(f"Starting MemoryGatedOptimizer loop | rounds={max_rounds} | poll={self.poll_interval}s")
        for rnd in range(1, max_rounds + 1):
            log.info(f"─── Round {rnd}/{max_rounds} ───────────────────────────")
            snap = self.profile_hardware()
            log.info(f"HW: {snap}")
            
            self.apply_elastic_quantization(snap)
            self.apply_thermal_pruning(snap)
            
            packet = self.export_weights_p2p()
            if callback:
                callback(snap, packet)
            
            if rnd < max_rounds:
                time.sleep(self.poll_interval)
        log.info("MemoryGatedOptimizer loop complete.")

    def __repr__(self) -> str:
        return (
            f"MemoryGatedOptimizer("
            f"node={self.node_id}, "
            f"precision={self.current_precision.value}, "
            f"vram_thresh={self.vram_threshold:.0%}, "
            f"temp_thresh={self.temp_threshold}°C, "
            f"arch={self.target_arch})"
        )
