import logging
import torch.nn as nn
import torch.nn.utils.prune as prune
from .datatypes import HardwareSnapshot

log = logging.getLogger("MemoryGatedOptimizer")

def compute_sparsity(model: nn.Module) -> float:
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
            zeros += (m.weight == 0).sum().item()
    return zeros / total if total > 0 else 0.0

def apply_thermal_pruning(
    model: nn.Module,
    snap: HardwareSnapshot,
    temp_threshold: float,
    prune_amount: float,
    pruned_layers: set
):
    if snap.gpu_temp_c <= temp_threshold:
        return

    log.warning(f"Thermal throttle detected! GPU {snap.gpu_temp_c:.0f}°C > "
                f"{temp_threshold:.0f}°C – pruning {prune_amount:.0%} of weights.")

    prunable = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]

    if not prunable:
        log.warning("No prunable layers found.")
        return

    prune.global_unstructured(
        prunable,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,
    )

    for module, param in prunable:
        key = f"{id(module)}_{param}"
        if key not in pruned_layers:
            try:
                prune.remove(module, param)
                pruned_layers.add(key)
            except ValueError:
                pass

    sparsity = compute_sparsity(model)
    log.info(f"Post-prune global sparsity: {sparsity:.2%}")
